import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))

import logging
from typing import Optional

import numpy as np
import pandas as pd

from data.data_sequence import generate_data_sequence
from data.preprocess import Preprocess

from .category_encoding import encode_stat_features
from .objects_grouping import ObjectsGrouping
from .pca_transformations import make_pca_transformations
from .static_sequence_split import generate_static_and_sequence_datasets

logger = logging.getLogger(__name__)

__all__ = ["apartment_buildings_period_anomaly_detection_pipeline"]


THRESHOLD = 0.25


def anomaly_detection_pipeline(
    data: pd.DataFrame,
    temperature: pd.DataFrame,
    buildings: pd.DataFrame,
    threshold: Optional[float] = None,
) -> pd.DataFrame:

    if threshold is None:
        threshold = THRESHOLD

    logging.info("Prefiltering data...")

    data = data[
        (data["Тип объекта"] == "Многоквартирный дом")
        & data["Текущее потребление, Гкал"].notnull()
    ]
    buildings = buildings[buildings["Тип Объекта"] == "Многоквартирный дом"]

    preprocess = Preprocess()
    data, buildings, temperature = preprocess.fit_transform(
        data, buildings, temperature
    )

    logging.info("Generating static dataset and consumption time series...")

    df = generate_data_sequence(data)
    df.fillna(0, inplace=True)
    df = df[(df == 0).sum(axis=1) < df.shape[1]]
    n_periods = df.shape[1]

    logging.info("Generating static dataset and consumption time series...")

    df_stat, df_seq = generate_static_and_sequence_datasets(df, buildings)
    df_seq.fillna(0, inplace=True)

    logging.info("Grouping static features...")

    grouping = ObjectsGrouping()
    df_stat = grouping.fit_transform(df_stat)

    logging.info("Encoding categories of static features...")

    stat_features = encode_stat_features(df_stat)

    logging.info("Calculating heating energy usage per square meter...")

    seq_features = (df_seq.T.values / df_stat["Общая площадь объекта"].values).T

    period_results = {}
    for period in range(n_periods):

        logging.info("Calculating Hotelling's T-squared" "& Q residuals...")

        df, X = make_pca_transformations(seq_features, stat_features, df_stat, period)

        logging.info("Generating anomaly detection dataframe...")

        result = apartment_buildings_period_anomaly_detection_df(
            df_seq, df_stat, df, X, period, threshold
        )
        period_results[period] = result

    return period_results


def apartment_buildings_period_anomaly_detection_df(
    df_seq: pd.DataFrame,
    df_stat: pd.DataFrame,
    df: pd.DataFrame,
    X: np.array,
    period: int,
    threshold: float,
) -> pd.DataFrame:
    """
    Генерирует pd.DataFrame с для многоквартирных домов, у которых в заданном периоде было  ненулевое
    потребления теплоенергии.

    На основании данных об удельном потреблении теплоэнергии на кв.метр площади, каждому объекту
    присваиваются значения Hotelling's T-squared" "и Q residuals, по которым можно определить,
    было ли потребление аномальным.

    Параллельно рассчитываются медианные значения удельного потребления теплоэнергии в разрезе групп:
    - год постройки (по группам до 1958 г., 1959-1989 гг., 1990-2000 гг., 2001-2010 гг., 2011-2024 гг.),
    - этажность (по группам 1-2 этажа, 3-4 этажа, 5-9 этажей,10-12 этажей, 13 и более этажей),
    - наличие ГВС ИТП (горячей воды, учитываемой тем же прибором).

    Аномально низкое/ высокое потребление (отклонение от медианы по группе более 25%) помечается
    в соответствующих колонках.
    """

    drop_list = ["Улица", "Дата постройки 2", "Группа общая площадь объекта"]
    cols = [col for col in df_stat.columns if col not in drop_list]

    t = df_seq.iloc[:, period]
    df.index = df_stat[t > 0].index
    result = pd.concat(
        [
            df_stat[cols],
            df_seq.iloc[:, period],
            pd.DataFrame(
                X[:, -1],
                index=df_stat[t > 0].index,
                columns=["Удельное потребление теплоэнергии на кв.метр площади"],
            ),
            df[["Hotelling's T-squared", "Q residuals"]],
        ],
        axis=1,
        join="inner",
    )
    group_ind = ["Группа год постройки", "Группа этажность объекта", "Вид энерг-а ГВС"]
    feature = "Удельное потребление теплоэнергии на кв.метр площади"
    medians = (
        result[group_ind + [feature]]
        .groupby(group_ind)[feature]
        .transform(lambda x: x.median())
    )
    result[f"ниже медианы"] = medians > result[feature]
    result[f"{threshold}% ниже медианы"] = medians > result[feature] / (1 - threshold)
    result[f"{threshold}% выше медианы"] = medians < result[feature] / (1 + threshold)

    return result

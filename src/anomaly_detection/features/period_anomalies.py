import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.data_sequence import generate_data_sequence
from data.preprocess import Preprocess

from .category_encoding import encode_stat_features
from .objects_grouping import ObjectsGrouping
from .pca_transformations import make_pca_transformations
from .static_sequence_split import generate_static_and_sequence_datasets

logger = logging.getLogger(__name__)

__all__ = ["apartment_buildings_period_anomaly_detection_pipeline"]


THRESHOLD = 0.25
ALPHA = 5
BETA = 95

PATH = ""
FOLDER = "data/06_model_output/"
RESULT_FOLDER = "results/4_period_anomalies/"
RESULT_FILE_NAMES = ["all_periods_anomalies.xlsx", "all_periods_anomalies_pivot.xlsx"]


def anomaly_detection_pipeline(
    data: pd.DataFrame,
    temperature: pd.DataFrame,
    buildings: pd.DataFrame,
    threshold: Optional[float] = None,
    alpha: Optional[int] = None,
    beta: Optional[int] = None,
    path: Optional[str] = None,
    folder: Optional[str] = None,
    result_folder: Optional[str] = None,
    result_file_names: Optional[list[str]] = None,
) -> pd.DataFrame:

    if threshold is None:
        threshold = THRESHOLD
    if alpha is None:
        alpha = ALPHA
    if beta is None:
        beta = BETA
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER
    if result_folder is None:
        result_folder = RESULT_FOLDER
    if result_file_names is None:
        result_file_names = RESULT_FILE_NAMES

    files_folder = f"{path}{folder}"
    result_file_names = [f"{path}{result_folder}{name}" for name in result_file_names]

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

    logging.info("Generating data sequence...")

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
    all_periods_anomalies = pd.DataFrame()
    for period in tqdm(range(n_periods)):

        logging.info("Calculating Hotelling's T-squared" "& Q residuals...")

        df, X = make_pca_transformations(seq_features, stat_features, df_stat, period)

        logging.info("Generating anomaly detection dataframe...")

        result = apartment_buildings_period_anomaly_detection_df(
            df_seq, df_stat, df, X, period, threshold
        )
        period_results[period] = result

        logging.info("Saving results...")

        t = result.columns[9]
        result.to_excel(f"{files_folder}{t.strftime('%Y-%m')}.xlsx")

        logging.info("Selecting anomalies...")

        cond1, cond2, cond3, cond4, cond5 = select_anomalies(
            result, alpha, beta, threshold
        )
        below_median = result[(cond1 | cond2) & cond4]
        above_median = result[(cond1 | cond2) & cond5]

        df1 = pd.concat([below_median, above_median], axis=0).rename(
            columns={t: "Текущее потребление, Гкал"}
        )
        df1["Период потребления"] = t
        all_periods_anomalies = pd.concat([all_periods_anomalies, df1], axis=0)

    logging.info("Saving dataframes with anomalies...")

    all_periods_anomalies.to_excel(result_file_names[0])

    anomalies_seq = all_periods_anomalies.reset_index().pivot_table(
        index="index", columns="Период потребления", values="Текущее потребление, Гкал"
    )
    drop_cols = ["Улица", "Дата постройки 2", "Группа общая площадь объекта"]
    cols = [i for i in df_stat.columns if i not in drop_cols]
    all_periods_anomalies_pivot = pd.concat(
        [df_stat[cols], anomalies_seq], axis=1, join="inner"
    )
    all_periods_anomalies_pivot.to_excel(result_file_names[1])

    return period_results, all_periods_anomalies, all_periods_anomalies_pivot


def select_anomalies(
    df: pd.DataFrame, alpha: int = 5, beta: int = 95, threshold: float = 0.25
) -> Tuple[bool, bool, bool, bool, bool]:
    """
    Генерирует условия для выбора аномалий.
    На входе получает:
      df -  датафрейм с предрассчитанными Hotelling's T-squared, Q residuals, индикаторами,
            является ли значиение удельного расхода теплоэнергии объекта на единицу площади
            1) ниже медианы по группе:
                - год постройки (по группам до 1958 г., 1959-1989 гг., 1990-2000 гг., 2001-2010 гг., 2011-2024 гг.),
                - этажность (по группам 1-2 этажа, 3-4 этажа, 5-9 этажей,10-12 этажей, 13 и более этажей),
                - наличие ГВС ИТП (горячей воды, учитываемой тем же прибором);
            2) ниже медианы по группе более чем на {threshold} (например, threshold = 0.25 - на 25%);
            3) выше медианы по группе более чем на {threshold}.
    alpha - {alpha} перцентиль ограничивает слева на кривой нормального распределения
            {alpha}% численности объектов с наименьшими значениями Hotelling's T-squared
     beta - {beta}% перцентиль ограничивает слева {beta}% численности объектов с наименьшими
            значениями Q residuals.
    threshold - точка отсечения выше/ ниже медианного значения удельного расхода теплоэнергии
            объекта на единицу площади, после которой значение считается аномально высоким/ низким.

    Выводит:
    cond1 - объекты, у которых значения Hotelling's T-squared попадают в {alpha} перцентиль.
    cond2 - объекты, у которых Q residuals не попадают в {beta} перцентиль.
            {beta}% объектов с наибольшими значениями Q residuals.
    cond3 - индикатор того, что удельный расход теплоэнергии объекта на единицу площади
            ниже медианы по группе.
    cond4 - индикатор того, что удельный расход теплоэнергии объекта на единицу площади
            ниже медианы по группе более, чем на {threshold}%
    cond5 - индикатор того, что удельный расход теплоэнергии объекта на единицу площади
            выше медианы по группе более, чем на {threshold}%
    """
    q1 = np.percentile(df["Hotelling's T-squared"], alpha)
    q2 = np.percentile(df["Q residuals"], beta)

    cond1 = df["Hotelling's T-squared"] < q1
    cond2 = df["Q residuals"] > q2
    cond3 = df["ниже медианы"] == True

    cond4 = df[f"{int(threshold * 100)}% ниже медианы"] == True
    cond5 = df[f"{int(threshold * 100)}% выше медианы"] == True

    return cond1, cond2, cond3, cond4, cond5


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
    result[f"{int(threshold * 100)}% ниже медианы"] = medians > result[feature] / (
        1 - threshold
    )
    result[f"{int(threshold * 100)}% выше медианы"] = medians < result[feature] / (
        1 + threshold
    )

    return result

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["generate_anomaly_detection_df"]

THRESHOLD = 0.25


def generate_anomaly_detection_df(
    df_seq: pd.DataFrame,
    df_stat: pd.DataFrame,
    df: pd.DataFrame,
    X: np.array,
    period: int,
    threshold: Optional[float] = None,
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

    if threshold is None:
        threshold = THRESHOLD

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

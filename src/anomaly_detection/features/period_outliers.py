import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))


import logging
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["find_period_outliers"]


SAVE = False

THRESHOLD = 0.25
N_PERIODS = 24
PATH = ""
FILE_NAMES = [
    "results/outliers_addr.pickle",
    "results/consumption_mask.xlsx",
]


def get_outlers(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    n_periods: Optional[int] = None,
    save: Optional[bool] = None,
    path: Optional[str] = None,
    file_name_outliers_addr: Optional[str] = None,
    file_name_consumption_mask: Optional[str] = None,
) -> Tuple[dict, pd.DataFrame]:
    """
    Для типов объекта «Многоквартирный дом» выявляет аномально низкое/высокое (отклонение более 25%)
    потребление объекта в конкретном месяце по сравнению с аналогичными объектами по критериям:

    - год постройки (по группам до 1958 г., 1959-1989 гг., 1990-2000 гг., 2001-2010 гг., 2011-2024 гг.),
    - этажность (по группам 1-2 этажа, 3-4 этажа, 5-9 этажей,10-12 этажей, 13 и более этажей),
    - площадь (±10%),
    - наличие ГВС ИТП (горячей воды, учитываемой тем же прибором).

    consumption_mask:
    - 1: аномально низкое потребление (отклонение более 25%)
    - 2: медианное потребление +/- 25%
    - 3: аномально высокое потребление (отклонение более 25%)

    """
    if threshold is None:
        threshold = THRESHOLD
    if n_periods is None:
        n_periods = N_PERIODS
    if save is None:
        save = SAVE
    if path is None:
        path = PATH
    if file_name_outliers_addr is None:
        file_name_outliers_addr = f"{path}{FILE_NAMES[0]}"
    if file_name_consumption_mask is None:
        file_name_underconsumption = f"{path}{FILE_NAMES[1]}"

    df.reset_index(drop=True, inplace=True)
    df.iloc[:, -n_periods:] = np.where(
        df.iloc[:, -n_periods:] == 0, np.nan, df.iloc[:, -n_periods:]
    )
    ratio = (df.iloc[:, -n_periods:].values.T / df["Общая площадь объекта"].values).T
    df_ratio = pd.concat(
        [df.iloc[:, :-n_periods], pd.DataFrame(ratio, columns=df.columns[-n_periods:])],
        axis=1,
    )
    outliers_addr = {}
    consumption_mask = df.copy()
    consumption_mask.iloc[:, -n_periods:] = np.nan

    group_ind = ["Группа год постройки", "Группа этажность объекта", "Вид энерг-а ГВС"]
    for period in df_ratio.columns[-n_periods:]:
        medians = (
            df_ratio[group_ind + [period]]
            .groupby(group_ind)[period]
            .transform(lambda x: x.median())
        )
        cond_under = medians > df_ratio[period] / (1 - threshold)
        cond_over = medians < df_ratio[period] / (1 + threshold)

        outliers_addr[period] = {
            "underconsumption": df_ratio[cond_under]["Адрес объекта 2"].tolist(),
            "overconsumption": df_ratio[cond_over]["Адрес объекта 2"].tolist(),
        }
        consumption_mask[period] = np.where(
            cond_under,
            1,
            np.where(cond_over, 3, np.where(df_ratio[period].notnull(), 2, np.nan)),
        )

    if save:
        with open(file_name_outliers_addr, "wb") as f:
            pickle.dump(outliers_addr, f)
        consumption_mask.to_excel(file_name_underconsumption)

    return outliers_addr, consumption_mask

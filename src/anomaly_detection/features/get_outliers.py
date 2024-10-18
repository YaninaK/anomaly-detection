import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SAVE = True

THRESHOLD = 0.25
N_PERIODS = 24
PATH = ""
FILE_NAMES = ["results/underconsumption.xlsx", "results/overconsumption.xlsx"]


def get_outlers(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    n_periods: Optional[int] = None,
    save: Optional[bool] = None,
    path: Optional[str] = None,
    file_name_underconsumption: Optional[str] = None,
    file_name_overconsumption: Optional[str] = None,
) -> Tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """
    Для типов объекта «Многоквартирный дом» выявляет аномально низкое/высокое (отклонение более 25%)
    потребление объекта в конкретном месяце по сравнению с аналогичными объектами по критериям:

    - год постройки (по группам до 1958 г., 1959-1989 гг., 1990-2000 гг., 2001-2010 гг., 2011-2024 гг.),
    - этажность (по группам 1-2 этажа, 3-4 этажа, 5-9 этажей,10-12 этажей, 13 и более этажей),
    - площадь (±10%),
    - наличие ГВС ИТП (горячей воды, учитываемой тем же прибором).

    """
    if threshold is None:
        threshold = THRESHOLD
    if n_periods is None:
        n_periods = N_PERIODS
    if save is None:
        save = SAVE
    if path is None:
        path = PATH
    if file_name_underconsumption is None:
        file_name_underconsumption = f"{path}{FILE_NAMES[0]}"
    if file_name_overconsumption is None:
        file_name_overconsumption = f"{path}{FILE_NAMES[1]}"

    ratio = (df.iloc[:, -n_periods:].values.T / df["Общая площадь объекта"].values).T
    df_ratio = pd.concat(
        [df.iloc[:, :-n_periods], pd.DataFrame(ratio, columns=df.columns[-n_periods:])],
        axis=1,
    )

    under_medians = {}
    over_medians = {}

    underconsumption = df.copy()
    overconsumption = df.copy()
    underconsumption.iloc[:, -n_periods:] = np.nan
    overconsumption.iloc[:, -n_periods:] = np.nan

    for period in df_ratio.columns[-n_periods:]:
        medians = df_ratio.groupby(
            ["Группа год постройки", "Группа этажность объекта", "Вид энерг-а ГВС"]
        )[period].transform(lambda x: x.median())

        under_medians_addr = df_ratio[medians > df_ratio[period] / (1 - threshold)][
            "Адрес объекта 2"
        ].tolist()
        over_medians_addr = df_ratio[medians < df_ratio[period] / (1 + threshold)][
            "Адрес объекта 2"
        ].tolist()

        under_medians[period] = under_medians_addr
        over_medians[period] = over_medians_addr

        underconsumption.loc[df["Адрес объекта 2"].isin(under_medians_addr), period] = (
            df.loc[df["Адрес объекта 2"].isin(under_medians_addr), period]
        )
        overconsumption.loc[df["Адрес объекта 2"].isin(over_medians_addr), period] = (
            df.loc[df["Адрес объекта 2"].isin(over_medians_addr), period]
        )
        if save:
            underconsumption.to_excel(file_name_underconsumption)
            overconsumption.to_excel(file_name_overconsumption)

    return under_medians, over_medians, underconsumption, overconsumption

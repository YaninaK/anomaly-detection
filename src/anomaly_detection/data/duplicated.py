import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["find_equal_values"]

SAVE = False
PATH = ""
FILE_NAME = "results/equal_values.xlsx"


def get_equal_values(
    data: pd.DataFrame,
    save: Optional[bool] = None,
    path: Optional[str] = None,
    file_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Выявляет равные значения показаний в течение нескольких расчетных периодов
    """
    if save is None:
        save = SAVE
    if path is None:
        path = PATH
    if file_name is None:
        file_name = f"{path}{FILE_NAME}"

    data.replace(0, np.nan, inplace=True)
    cond1 = data["Текущее потребление, Гкал"].notnull()
    cond2 = (
        data[cond1]
        .groupby(["Адрес объекта 2"])["Текущее потребление, Гкал"]
        .transform(lambda x: x.duplicated(keep=False))
    )
    equal_values = (
        data[cond1 & cond2]
        .sort_values(["Адрес объекта 2", "Период потребления"])
        .iloc[:, :-1]
    )
    if save:
        equal_values.to_excel(file_name)

    return equal_values

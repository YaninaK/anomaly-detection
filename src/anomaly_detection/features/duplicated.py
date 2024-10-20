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
    у объекта с индексом:
    [
        "Адрес объекта",
        "Тип объекта",
        "№ ОДПУ",
        "Вид энерг-а ГВС",
    ]
    """
    if save is None:
        save = SAVE
    if path is None:
        path = PATH
    if file_name is None:
        file_name = f"{path}{FILE_NAME}"

    data["Текущее потребление, Гкал"].replace(0, np.nan, inplace=True)
    combined_index = [
        "Адрес объекта",
        "Тип объекта",
        "№ ОДПУ",
        "Вид энерг-а ГВС",
    ]
    cond1 = data["Текущее потребление, Гкал"].notnull()
    cond2 = (
        data[cond1]
        .groupby(combined_index)["Текущее потребление, Гкал"]
        .transform(lambda x: x.duplicated(keep=False))
    )
    equal_values = (
        data[cond1 & cond2]
        .sort_values(["Адрес объекта 2", "Период потребления"])
        .iloc[:, :-1]
    )
    equal_values["Вид энерг-а ГВС"] = np.where(
        equal_values["Вид энерг-а ГВС"] == 1, "ГВС-ИТП", None
    )

    if save:
        equal_values.to_excel(file_name, index=False)

    return equal_values

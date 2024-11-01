import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))

import logging
from typing import Optional

import numpy as np
import pandas as pd
from data.preprocess import Preprocess

logger = logging.getLogger(__name__)

__all__ = ["equal_values_identification"]


SAVE = False
PATH = ""
FOLDER = "results/2_equal_records/"
FILE_NAME = "equal_values.xlsx"


def equal_values_identification_pipeline(
    data: pd.DataFrame,
    buildings: pd.DataFrame,
    temperature: pd.DataFrame,
    save: Optional[bool] = None,
    path: Optional[str] = None,
    folder: Optional[str] = None,
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
    if folder is None:
        folder = FOLDER
    if file_name is None:
        file_name = FILE_NAME

    file_name = f"{path}{folder}{file_name}"

    logging.info("Prefiltering data...")

    preprocess = Preprocess()
    data, buildings, temperature = preprocess.fit_transform(
        data, buildings, temperature
    )

    logging.info("Idenfifying equal values...")

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
        logging.info("Saving dataframes with equal values...")

        equal_values.to_excel(file_name, index=False)

    return equal_values

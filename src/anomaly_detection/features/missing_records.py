import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))


import logging
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from data.data_sequence import generate_data_sequence
from data.preprocess import Preprocess

logger = logging.getLogger(__name__)

__all__ = ["find_missing_records"]


ALL_PERIODS = False
SAVE = True

PATH = ""
FOLDER = "results/1_missing_records/"
FILE_NAMES = [
    "missing_records.xlsx",
    "uninvoiced_objects.xlsx",
    "nonunique_objects.xlsx",
]


def identify_missing_data_and_nonunique_objects(
    data: pd.DataFrame,
    buildings: pd.DataFrame,
    temperature: pd.DataFrame,
    all_periods: Optional[bool] = None,
    save: Optional[bool] = None,
    path: Optional[str] = None,
    folder: Optional[str] = None,
    file_names: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if all_periods is None:
        all_periods = ALL_PERIODS
    if save is None:
        save = SAVE
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER
    if file_names is None:
        file_names = FILE_NAMES

    file_names = [f"{path}{folder}{name}" for name in file_names]

    logging.info("Prefiltering data...")

    preprocess = Preprocess()
    data, buildings, temperature = preprocess.fit_transform(
        data, buildings, temperature
    )

    logging.info("Generating data sequence...")

    df = generate_data_sequence(data)

    logging.info("Identifying missing records...")

    missing_consumption_records = identify_missing_records(
        df, all_periods, save, file_name=file_names[0]
    )

    logging.info("Identifying uninvoiced objects...")

    uninvoiced_objects = identify_uninvoiced_objects(
        df, buildings, save, file_name=file_names[1]
    )

    logging.info("Identifying nonunique objects...")

    nonunique_objects = identify_nonunique_objects(
        buildings, save, file_name=file_names[2]
    )

    return missing_consumption_records, uninvoiced_objects, nonunique_objects


def identify_missing_records(
    df: pd.DataFrame,
    all_periods: bool,
    save: bool,
    file_name: str,
) -> pd.DataFrame:

    if not all_periods:
        periods = [x for x in df.columns if x.month not in range(5, 10)]
        df = df.loc[:, periods]

    cond = (df.isnull() | (df == 0)).sum(axis=1) > 0
    missing_records = df[cond]

    if save:
        missing_records_result = missing_records.reset_index().drop(
            "Адрес объекта 2", axis=1
        )
        missing_records_result["Вид энерг-а ГВС"] = np.where(
            missing_records_result["Вид энерг-а ГВС"] == 1, "ГВС-ИТП", None
        )
        missing_records_result.to_excel(file_name, index=False)

    return missing_records


def identify_uninvoiced_objects(
    df: pd.DataFrame, buildings: pd.DataFrame, save: bool, file_name: str
) -> pd.DataFrame:
    """
    Выявляет объекты, по которым нет данных учета теплоэнергии.
    """

    df1 = df.reset_index()
    uninvoiced_objects = sorted(
        list(
            set(zip(buildings["Адрес объекта 2"], buildings["Тип Объекта"]))
            - set(zip(df1["Адрес объекта 2"], df1["Тип объекта"]))
        )
    )
    merge_basis = ["Адрес объекта 2", "Тип Объекта"]
    uninvoiced_objects = pd.DataFrame(uninvoiced_objects, columns=merge_basis).merge(
        buildings, how="left", on=merge_basis
    )[buildings.columns]

    if save:
        uninvoiced_objects.iloc[:, :-1].to_excel(file_name, index=False)

    return uninvoiced_objects


def identify_nonunique_objects(
    buildings: pd.DataFrame,
    save: bool,
    file_name: str,
) -> pd.DataFrame:
    """
    Выбирает неуникальные адреса объектов в разрезе типов объектов.
    """

    nonunique = buildings[
        buildings.duplicated(subset=["Адрес объекта", "Тип Объекта"], keep=False)
    ]
    if save:
        nonunique.iloc[:, :-1].to_excel(file_name, index=False)

    return nonunique

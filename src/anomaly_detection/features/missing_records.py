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

__all__ = ["find_missing_records"]


ALL_PERIODS = False
SAVE = False
PATH = ""
FILE_NAMES = [
    "results/missing_records.xlsx",
    "results/uninvoiced_objects.xlsx",
]


def select_missing_records(
    df: pd.DataFrame,
    all_periods: Optional[bool] = None,
    save: Optional[bool] = None,
    path: Optional[str] = None,
    file_name: Optional[str] = None,
) -> pd.DataFrame:

    if all_periods is None:
        all_periods = ALL_PERIODS
    if save is None:
        save = SAVE
    if path is None:
        path = PATH
    if file_name is None:
        file_name = f"{path}{FILE_NAMES[0]}"

    if not all_periods:
        periods = [x for x in df.columns if x.month not in range(5, 10)]
        df = df.loc[:, periods]

    cond = (df.isnull() | (df == 0)).sum(axis=1) > 0
    missing_records = df[cond]

    if save:
        missing_records.reset_index().drop("Адрес объекта 2", axis=1).to_excel(
            file_name, index=False
        )

    return missing_records


def select_uninvoiced_objects(
    df: pd.DataFrame,
    buildings: pd.DataFrame,
    save: Optional[bool] = None,
    path: Optional[str] = None,
    file_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Выявляет объекты, по которым нет данных учета теплоэнергии.
    """
    if save is None:
        save = SAVE
    if path is None:
        path = PATH
    if file_name is None:
        file_name = f"{path}{FILE_NAMES[1]}"

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

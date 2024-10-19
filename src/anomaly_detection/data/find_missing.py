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

SAVE = False
PATH = ""
FILE_NAMES = [
    "results/missing_records_addr.pickle",
    "results/missing_records.xlsx",
    "results/uninvoiced_buildings.xlsx",
]


def find_missing_records(
    data: pd.DataFrame,
    save: Optional[bool] = None,
    path: Optional[str] = None,
    file_name_addr: Optional[str] = None,
    file_name_df: Optional[str] = None,
) -> Tuple[dict, pd.DataFrame]:
    """
    Выявляет нулевые значения показаний за тепловую энергию в отопительный период (октябрь-апрель).
    Возвращает:
    missing_records_addr: dict  - словарь со списком адресов и типов объектов за каждый период
    missing_records_df: pd.DataFrame - датафрейм с адресами, типами объектов и пропусками за каждый период.

    """
    if save is None:
        save = SAVE
    if path is None:
        path = PATH
    if file_name_addr is None:
        file_name_addr = f"{path}{FILE_NAMES[0]}"
    if file_name_df is None:
        file_name_df = f"{path}{FILE_NAMES[1]}"

    cond = data["Период потребления"].apply(lambda x: x.month not in range(5, 10))
    df = data[cond].pivot_table(
        index=["Адрес объекта 2", "Тип объекта"],
        columns="Период потребления",
        values="Текущее потребление, Гкал",
    )
    df.replace(0, np.nan, inplace=True)
    missing_records_df = df[df.isnull().sum(axis=1) > 0]

    missing_records_addr = {}
    for period in missing_records_df.columns:
        missing_records_addr[period] = [
            (address, object_type)
            for (address, object_type) in missing_records_df.loc[
                missing_records_df[period].isnull(), period
            ].index
        ]

    missing_records_df.reset_index(inplace=True)

    if save:
        with open(file_name_addr, "wb") as f:
            pickle.dump(file_name_addr, f)
        missing_records_df.to_excel(file_name_df)

    return missing_records_addr, missing_records_df


def get_uninvoiced_buildings(
    data: pd.DataFrame,
    buildings: pd.DataFrame,
    save: Optional[bool] = None,
    path: Optional[str] = None,
    file_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Выявляет объекты без данных учета теплоэнергии в разрезе типов объектов.
    """
    if save is None:
        save = SAVE
    if path is None:
        path = PATH
    if file_name is None:
        file_name = f"{path}{FILE_NAMES[2]}"

    data_addr = (
        data.groupby(["Тип объекта", "Адрес объекта 2"])["Текущее потребление, Гкал"]
        .sum()
        .reset_index()
    )
    df = buildings.merge(
        data_addr,
        how="left",
        left_on=["Тип Объекта", "Адрес объекта 2"],
        right_on=["Тип объекта", "Адрес объекта 2"],
    )
    uninvoiced_buildings = df[df["Тип объекта"].isnull()].iloc[:, :-2]

    if save:
        uninvoiced_buildings.to_excel(file_name)

    return uninvoiced_buildings

import datetime
import logging
import os
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ["load_data"]

MONTH_DICT = {
    "Январь": 1,
    "Февраль": 2,
    "Март": 3,
    "Апрель": 4,
    "Май": 5,
    "Июнь": 6,
    "Июль": 7,
    "Август": 8,
    "Сентябрь": 9,
    "Октябрь": 10,
    "Ноябрь": 11,
    "Ноябрь": 11,
    "Декабрь": 12,
}
SAVE = True

PATH = ""
FILE_NAME = "data/02_intermediate/data.parquet.gzip"


def load_data(
    folder_path,
    save: Optional[bool] = None,
    month_dict: Optional[dict] = None,
    path: Optional[str] = None,
    file_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if save is None:
        save = SAVE
    if month_dict is None:
        month_dict = MONTH_DICT
    if path is None:
        path = PATH
    if file_name is None:
        file_name = f"{path}{FILE_NAME}"

    if save:
        data_files = [
            f
            for f in os.listdir(folder_path)
            if f[0] != "." and (f.split()[1] in ["2021", "2022", "2023"])
        ]
        data = pd.DataFrame()

        for name in tqdm(data_files):
            month, year, _ = name.split()
            df = pd.read_excel(f"{folder_path}{name}", skiprows=1)
            df["Период потребления"] = pd.to_datetime(f"{year}-{month_dict[month]}")
            data = pd.concat([data, df], axis=0)

        data = data[data["Вид энерг-а ГВС"] != "ГВС (централ)"].reset_index()
        data.to_parquet(file_name, compression="gzip")
    else:
        data = pd.read_parquet(file_name)

    temperature = (
        pd.read_excel(
            f"{folder_path}Температуры, продолжительность ОП.xls",
            index_col=1,
            skiprows=1,
        )
        .T[1:]
        .reset_index()
    )

    buildings = pd.read_excel(
        f"{folder_path}Тип строения, этажность, площадь, год постройки.xlsx", skiprows=1
    )

    return data, temperature, buildings

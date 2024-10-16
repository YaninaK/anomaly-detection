import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))


import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["find_missing_records"]

SAVE = False
PATH = ""
FILE_NAME = "results/missing_data.xlsx"


def find_missing_records(
    data: pd.DataFrame,
    save: Optional[bool] = None,
    path: Optional[str] = None,
    file_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Выявляет нулевые значения показаний за тепловую энергию в отопительный период (октябрь-апрель)
    """
    if save is None:
        save = SAVE
    if path is None:
        path = PATH
    if file_name is None:
        file_name = f"{path}{FILE_NAME}"

    cond1 = data["Период потребления"].apply(lambda x: x.month not in range(5, 10))
    df = data[cond1].pivot_table(
        index="Адрес объекта",
        columns="Период потребления",
        values="Текущее потребление, Гкал",
    )
    df.replace(0, np.nan, inplace=True)

    r_records = df.count(axis=1)
    address_missing_values = r_records[r_records < len(df.columns)].index
    cond2 = data["Адрес объекта"].isin(address_missing_values)

    df1 = data[cond1 & cond2].groupby(["Период потребления", "Адрес объекта"]).last()
    df2 = df.loc[address_missing_values].unstack()

    cols = [col for col in data.columns[1:] if col != "Дата текущего показания"]
    missing_data = (
        pd.concat([df1, df2], axis=1)
        .reset_index()[cols]
        .sort_values(["Адрес объекта", "Период потребления"], ascending=[True, False])
        .reset_index(drop=True)
    )[cols]

    for col in [
        "Подразделение",
        "№ ОДПУ",
        "Вид энерг-а ГВС",
        "Тип объекта",
        "Адрес объекта 2",
    ]:
        missing_data[col] = missing_data.groupby("Адрес объекта")[col].transform(
            lambda x: x.ffill().bfill()
        )

    missing_data.loc[
        missing_data["Текущее потребление, Гкал"] == 0, "Текущее потребление, Гкал"
    ] = np.nan

    if save:
        missing_data.iloc[:, :-1].to_excel(file_name)

    return missing_data

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
    "results/uninvoiced_buildings.xlsx",
]


def get_missing_records(
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
        missing_records.to_excel(file_name)

    return missing_records

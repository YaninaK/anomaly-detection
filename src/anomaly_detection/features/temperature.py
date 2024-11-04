import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))

import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data import N_TEST_PERIODS, N_VALID_PERIODS

logger = logging.getLogger(__name__)

__all__ = ["transform_temperature"]


PATH = ""
FOLDER = "data/04_feature/"
FILE_NAME = "temperature_scaler.pkl"


def transform_temperature(
    temperature: pd.DataFrame,
    n_valid_periods: Optional[int] = None,
    n_test_periods: Optional[int] = None,
    path: Optional[str] = None,
    folder: Optional[str] = None,
    file_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Трансформирует температуру в диапазоне от 0 до 1.
    Сохраняет обученный scaler.
    """
    if n_valid_periods is None:
        n_valid_periods = N_VALID_PERIODS
    if n_test_periods is None:
        n_test_periods = N_TEST_PERIODS
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER
    if file_name is None:
        file_name = FILE_NAME

    file_name = f"{path}{folder}{file_name}"

    t = n_valid_periods + n_test_periods

    temperature["t_scaled"] = np.nan
    t_ = temperature["Тн.в, град.С"].values.reshape(-1, 1)

    temperature_scaler = MinMaxScaler()
    temperature.iloc[:-t, -1] = temperature_scaler.fit_transform(t_[:-t])
    temperature.iloc[-t:, -1] = temperature_scaler.transform(t_[-t:])

    with open(file_name, "wb") as f:
        pickle.dump(temperature_scaler, f)

    return temperature

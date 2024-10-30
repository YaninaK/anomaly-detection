import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))

import logging
from typing import Optional, Tuple

import pandas as pd

from models import AUTOENCODER_CONFIG as CONFIG

from . import N_TEST_PERIODS, N_VALID_PERIODS

logger = logging.getLogger(__name__)

__all__ = ["generate_data_sequence"]


def generate_data_sequence(data: pd.DataFrame) -> pd.DataFrame:
    """
    Создает последовательность данных без пропусков.
    """
    combined_index = sorted(
        list(
            set(
                zip(
                    data["Адрес объекта"],
                    data["Тип объекта"],
                    data["№ ОДПУ"],
                    data["Вид энерг-а ГВС"],
                    data["Адрес объекта 2"],
                )
            )
        )
    )
    merge_basis = [
        "Адрес объекта",
        "Тип объекта",
        "№ ОДПУ",
        "Вид энерг-а ГВС",
        "Адрес объекта 2",
    ]
    df = pd.DataFrame(combined_index, columns=merge_basis)
    periods = sorted(data["Период потребления"].unique().tolist())
    for period in periods:
        current_period = (
            data[data["Период потребления"] == period][
                merge_basis + ["Текущее потребление, Гкал"]
            ]
            .rename(columns={"Текущее потребление, Гкал": period})
            .groupby(merge_basis, as_index=False)[period]
            .sum()
        )
        df = df.merge(current_period, how="left", on=merge_basis)

    df.set_index(merge_basis, inplace=True)

    return df


def sequence_train_validation_split(
    df_seq: pd.DataFrame,
    n_valid_periods: Optional[int] = None,
    n_test_periods: Optional[int] = None,
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Делит временной ряд на обучающую, валидационную и тестовую выборки
    по числу периодов, оставленных для валидации и тестирования, и длине
    последовательностей, на которые делится временной ряд.
    """
    if n_valid_periods is None:
        n_valid_periods = N_VALID_PERIODS
    if n_test_periods is None:
        n_test_periods = N_TEST_PERIODS
    if config is None:
        config = CONFIG

    train = df_seq.iloc[:, : -n_test_periods - n_valid_periods]
    valid = df_seq.iloc[
        :,
        -n_test_periods - n_valid_periods - config["seq_length"] + 1 : -n_test_periods,
    ]
    test = df_seq.iloc[:, -n_test_periods - config["seq_length"] + 1 :]

    return train, valid, test

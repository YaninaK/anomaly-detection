import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from models import AUTOENCODER_CONFIG as CONFIG

logger = logging.getLogger(__name__)

__all__ = ["generate_static_and_sequence_datasets"]


def generate_static_and_sequence_datasets(
    df: pd.DataFrame,
    temperature: pd.DataFrame,
    buildings: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. Соединяет данные по объектам с данными по потреблению теплоэнергии.
    2. Удаляет объекты с нулевым потреблением во всех периодах.
    3. Генерирует 2 датасета:
        df_stat - на датасет со статичными данными
        df_seq - датасет с временными рядами.
    """
    if config is None:
        config = CONFIG

    n_periods = config["n_periods"]
    df.iloc[:, -n_periods:] = np.where(
        df.iloc[:, -n_periods:] == 0, np.nan, df.iloc[:, -n_periods:]
    )

    logging.info("Merging consumption data sequence with objects dataset...")

    df_comb = buildings.merge(
        df.reset_index(),
        left_on=["Тип Объекта", "Адрес объекта 2"],
        right_on=["Тип объекта", "Адрес объекта 2"],
        how="right",
    ).drop_duplicates(
        subset=["Адрес объекта_y", "Тип объекта", "№ ОДПУ", "Вид энерг-а ГВС"],
        keep=False,
    )
    df_comb = df_comb[df_comb["Адрес объекта_x"].notnull()].reset_index(drop=True)

    logging.info("Splitting combined dataset into static and sequence dataset...")

    df_stat = df_comb[
        [
            "Адрес объекта 2",
            "Тип объекта",
            "№ ОДПУ",
            "Вид энерг-а ГВС",
            "Этажность объекта",
            "Дата постройки",
            "Общая площадь объекта",
        ]
    ]
    df_seq = df_comb.iloc[:, -n_periods:]
    return df_stat, df_seq

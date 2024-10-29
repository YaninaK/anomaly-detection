import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))

import logging
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from data.data_sequence import generate_data_sequence
from data.preprocess import Preprocess
from features.model_inputs import Generator
from features.objects_grouping import ObjectsGrouping

from . import AUTOENCODER_CONFIG as CONFIG

logger = logging.getLogger(__name__)

__all__ = ["preprocess_data"]


PATH = ""
FOLDER = "data/04_feature/"
TEMPERATURE_SCALER_FILE_NAME = "temperature_scaler.pkl"
GENERATOR_TRAIN_DF_FILE_NAME = "train_df.parquet.gzip"
GENERATOR_VALID_DF_FILE_NAME = "valid_df.parquet.gzip"


def data_preprocessing_pipeline(
    data: pd.DataFrame,
    temperature: pd.DataFrame,
    buildings: pd.DataFrame,
    config: Optional[dict] = None,
    path: Optional[str] = None,
    folder: Optional[str] = None,
    temperature_scaler_file_name: Optional[str] = None,
    generator_train_df_file_name: Optional[str] = None,
    generator_valid_df_file_name: Optional[str] = None,
) -> Tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    tf.data.Dataset,
    tf.data.Dataset,
]:
    """
    Конвейер подготовки данных для автоэнкодера.
    """

    if config is None:
        config = CONFIG
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER
    if temperature_scaler_file_name is None:
        temperature_scaler_file_name = TEMPERATURE_SCALER_FILE_NAME
    if generator_train_df_file_name is None:
        generator_train_df_file_name = GENERATOR_TRAIN_DF_FILE_NAME
    if generator_valid_df_file_name is None:
        generator_valid_df_file_name = GENERATOR_VALID_DF_FILE_NAME

    temperature_scaler_file_name = f"{path}{folder}{temperature_scaler_file_name}"

    logging.info("Prefiltering data...")

    preprocess = Preprocess()
    data, buildings, temperature = preprocess.fit_transform(
        data, buildings, temperature
    )

    logging.info("Generating consumption data sequence...")

    df = generate_data_sequence(data)
    n_periods = config["n_periods"]
    df.iloc[:, -n_periods:] = np.where(
        df.iloc[:, -n_periods:] == 0, np.nan, df.iloc[:, -n_periods:]
    )
    df /= temperature["Число дней"]

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
    df_seq = df_comb.iloc[:, -n_periods:] / temperature["Число дней"]

    logging.info(
        "Splitting sequence dataset into train, validation and test datasets..."
    )

    test_periods = 4
    validation_period = 1
    train = df_seq.iloc[:, : -test_periods - validation_period]
    valid = df_seq.iloc[
        :, -test_periods - validation_period - config["seq_length"] + 1 : -test_periods
    ]
    test = df_seq.iloc[:, -test_periods - config["seq_length"] + 1 :]

    logging.info("Normalizing temperarure...")

    temperature["t_scaled"] = np.nan
    t_ = temperature["Тн.в, град.С"].values.reshape(-1, 1)
    temperature_scaler = MinMaxScaler()
    temperature.iloc[:-test_periods, -1] = temperature_scaler.fit_transform(
        t_[:-test_periods]
    )
    temperature.iloc[-test_periods:, -1] = temperature_scaler.transform(
        t_[-test_periods:]
    )
    with open(temperature_scaler_file_name, "wb") as f:
        pickle.dump(temperature_scaler, f)

    logging.info("Grouping static features...")

    grouping = ObjectsGrouping()
    df_stat = grouping.fit_transform(df_stat)

    logging.info("Generating tensorflow datasets for autoencoder model training ...")

    generator = Generator()
    ds_train, train_df = generator.fit_transform(
        train, temperature, df_stat, path, file_name=generator_train_df_file_name
    )
    ds_valid, valid_df = generator.fit_transform(
        valid, temperature, df_stat, path, file_name=generator_valid_df_file_name
    )

    return ds_train, ds_valid, train_df, valid_df, df_seq, temperature, df_stat

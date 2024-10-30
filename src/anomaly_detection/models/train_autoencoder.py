import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from data.data_sequence import generate_data_sequence, sequence_train_validation_split
from data.preprocess import Preprocess
from features.model_inputs import Generator
from features.objects_grouping import ObjectsGrouping
from features.static_sequence_split import generate_static_and_sequence_datasets
from features.temperature import transform_temperature

from . import AUTOENCODER_CONFIG as CONFIG

logger = logging.getLogger(__name__)

__all__ = ["preprocess_data"]


PATH = ""
FOLDER = "data/04_feature/"
TRAIN_DF_FILE_NAME = "train_df.parquet.gzip"
VALID_DF_FILE_NAME = "valid_df.parquet.gzip"


def data_preprocessing_pipeline(
    data: pd.DataFrame,
    temperature: pd.DataFrame,
    buildings: pd.DataFrame,
    config: Optional[dict] = None,
    path: Optional[str] = None,
    folder: Optional[str] = None,
    train_df_file_name: Optional[str] = None,
    valid_df_file_name: Optional[str] = None,
) -> Tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
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
    if train_df_file_name is None:
        train_df_file_name = TRAIN_DF_FILE_NAME
    if valid_df_file_name is None:
        valid_df_file_name = VALID_DF_FILE_NAME

    logging.info("Prefiltering data...")

    preprocess = Preprocess()
    data, buildings, temperature = preprocess.fit_transform(
        data, buildings, temperature
    )

    logging.info("Generating static dataset and consumption time series...")

    df = generate_data_sequence(data)
    df_stat, df_seq = generate_static_and_sequence_datasets(df, temperature, buildings)

    logging.info(
        "Splitting sequence dataset into train, validation and test datasets..."
    )
    train, valid, test = sequence_train_validation_split(df_seq)

    logging.info("Transforming temperature...")

    temperature = transform_temperature(temperature, path=path)

    logging.info("Grouping static features...")

    grouping = ObjectsGrouping()
    df_stat = grouping.fit_transform(df_stat)

    logging.info("Generating tensorflow datasets for autoencoder model training ...")

    generator = Generator(model_type=config["model_type"])
    ds_train, train_df = generator.fit_transform(
        train, temperature, df_stat, path, file_name=train_df_file_name
    )
    ds_valid, valid_df = generator.fit_transform(
        valid, temperature, df_stat, path, file_name=valid_df_file_name
    )

    return ds_train, ds_valid, train_df, valid_df, df_seq, temperature, df_stat

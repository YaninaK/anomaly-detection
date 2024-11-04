import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from features.model_inputs import Generator
from models.evaluation import evaluate_autoencoder_estimations

from . import ETHALON_MODEL_CONFIG as CONFIG

logger = logging.getLogger(__name__)

__all__ = ["preprocess_data_ethalon_model"]


BATCH_SIZE = 128


def ethalon_model_data_preprocessing_pipeline(
    lstm_model: tf.keras.Model,
    ds_train: tf.data.Dataset,
    train_df: pd.DataFrame,
    batch_size: Optional[int] = None,
    config: Optional[dict] = None,
) -> Tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if batch_size is None:
        batch_size = BATCH_SIZE
    if config is None:
        config = CONFIG

    logging.info("Evaluating autoencoder estimations...")

    train_df = evaluate_autoencoder_estimations(
        lstm_model, ds_train, train_df, batch_size
    )

    logging.info("Selecting data for ethalon model...")

    selected_inputs_df = train_df[
        train_df["Индекс соответствия прогнозу"] < config["quantile"]
    ]

    logging.info(
        "Splitting selected_inputs for ethalon model into train and validation datasets..."
    )
    train_df_et, valid_df_et = ethalon_model_train_validation_split(selected_inputs_df)

    logging.info("Generating tensorflow datasets for ethalon model training ...")

    generator = Generator(model_type=config["model_type"])
    ds_train_et = generator.get_tf_dataset(train_df_et)
    ds_valid_et = generator.get_tf_dataset(valid_df_et)

    return (
        ds_train_et,
        ds_valid_et,
        train_df_et,
        valid_df_et,
        train_df,
        selected_inputs_df,
    )


def ethalon_model_train_validation_split(
    selected_inputs_df: pd.DataFrame, valid_df_frac: float = 0.2, random_state: int = 25
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разбивает последовательности, отобранные для эталонной модели на обучающую и тестовую
    выборки.
    """
    valid_df_et = selected_inputs_df.sample(
        frac=valid_df_frac, random_state=random_state
    )
    train_ind = list(set(selected_inputs_df.index) - set(valid_df_et.index))
    train_df_et = selected_inputs_df.loc[train_ind, :]

    return train_df_et, valid_df_et

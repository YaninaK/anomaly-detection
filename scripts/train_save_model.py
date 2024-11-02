#!/usr/bin/env python3
"""Train and save model for Anomaly detection"""

import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))

import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from src.anomaly_detection.data.make_dataset import load_data
from src.anomaly_detection.models.LSTM_model import get_model
from src.anomaly_detection.models.serialize import store
from src.anomaly_detection.models.train_autoencoder import \
    data_preprocessing_pipeline
from src.anomaly_detection.models.train_ethalon_model import \
    ethalon_model_data_preprocessing_pipeline

logger = logging.getLogger()


METRICS = [
    tf.keras.metrics.MeanAbsoluteError(name="mae"),
    tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
    tf.keras.metrics.MeanSquaredError(name="mse"),
    tf.keras.metrics.RootMeanSquaredError(name="rmse"),
]
BATCH_SIZE = 128


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-d",
        "--data_folder_path",
        required=False,
        default="data/01_raw/",
        help="consumption, objects and temperature datasets store folder path",
    )
    argparser.add_argument(
        "-o1",
        "--output_autoencoder",
        required=False,
        default=["autoencoder_v1.keras", "autoencoder_training_history_v1.joblib"],
        help="filenames to store autoencoder and autoencoder training history",
    )
    argparser.add_argument(
        "-o2",
        "--output_ethalon_model",
        required=False,
        default=["ethalon_model_v1.keras", "ethalon_model_training_history_v1.joblib"],
        help="filenames to store ethalon model and ethalon model training history",
    )
    argparser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Reading data...")

    data, temperature, buildings = load_data(args.data_folder_path)

    logging.info("Preprocessing data...")
    ds_train, ds_valid, train_df, valid_df, df_seq, temperature, df_stat = (
        data_preprocessing_pipeline(data, temperature, buildings)
    )
    logging.info("Training autoencoder...")
    train_store_autoencoder(
        df_stat, ds_train, ds_valid, args.output_autoencoder, batch_size=BATCH_SIZE
    )

    logging.info("Generating ethalon model inputs...")

    (
        ds_train_et,
        ds_valid_et,
        train_df_et,
        valid_df_et,
        train_df,
        selected_inputs_df,
    ) = ethalon_model_data_preprocessing_pipeline(
        lstm_model, ds_train, train_df, batch_size=BATCH_SIZE
    )

    logging.info("Training ethalon model...")

    train_store_ethalon_model()


def train_store_autoencoder(
    df_stat: pd.DataFrame,
    ds_train: tf.data.Dataset,
    ds_valid: tf.data.Dataset,
    file_names: list[str],
    batch_size: int,
    n_epochs: int = 200,
    steps_per_epoch: int = 239,
):
    autoencoder = get_model(df_stat, model_type="autoencoder")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.Huber(),
        metrics=METRICS,
    )
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 3e-2 * 0.99**epoch
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        min_delta=1e-06,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )
    autoencoder_history = autoencoder.fit(
        ds_train.shuffle(5000).batch(batch_size),
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_valid.batch(batch_size),
        batch_size=batch_size,
        verbose=1,
        callbacks=[reduce_lr, early_stopping],
        shuffle=True,
    )
    store(autoencoder, autoencoder_history, file_names)


def train_store_ethalon_model(
    selected_inputs_df: pd.DataFrame,
    ds_train_et: tf.data.Dataset,
    ds_valid_et: tf.data.Dataset,
    file_names: list[str],
    batch_size: int,
    n_epochs: int = 200,
    steps_per_epoch: int = 239,
):
    ethalon_model = get_model(selected_inputs_df, model_type="ethalon_model")
    ethalon_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.Huber(),
        metrics=METRICS,
    )
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 3e-2 * 0.99**epoch
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        min_delta=1e-06,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )
    ethalon_model_history = ethalon_model.fit(
        ds_train_et.shuffle(5000).batch(batch_size),
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_valid_et.batch(batch_size),
        batch_size=batch_size,
        verbose=1,
        callbacks=[reduce_lr, early_stopping],
        shuffle=True,
    )
    store(ethalon_model, ethalon_model_history, file_names)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e)
        sys.exit(1)

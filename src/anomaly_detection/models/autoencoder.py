import logging
from typing import Optional

import pandas as pd
import tensorflow as tf

from . import AUTOENCODER_CONFIG

logger = logging.getLogger(__name__)

__all__ = ["define_autoencoder"]


def get_autoencoder(
    df_stat: pd.DataFrame,
    config: Optional[dict] = None,
) -> tf.keras.Model:
    """
    Генерирует автоэнкодер для детектирования аномалий.

    """
    if config is None:
        config = AUTOENCODER_CONFIG

    preprocessing_layers = []
    for feature in config["features"][:2]:
        normalize_layer = tf.keras.layers.Normalization(axis=None)
        normalize_layer.adapt(df_stat[feature].fillna(0).values)
        preprocessing_layers.append(normalize_layer)

    integerlookup_layer = tf.keras.layers.IntegerLookup(output_mode="one_hot")
    integerlookup_layer.adapt(df_stat[config["features"][2]])
    preprocessing_layers.append(integerlookup_layer)

    for feature in config["features"][3:-1]:
        stringlookup_layer = tf.keras.layers.StringLookup(output_mode="one_hot")
        stringlookup_layer.adapt(df_stat[feature])
        preprocessing_layers.append(stringlookup_layer)

    inputs = [
        tf.keras.Input(shape=(1,), dtype=int, name=f"n_floors"),
        tf.keras.Input(shape=(1,), dtype=float, name=f"area"),
        tf.keras.Input(shape=(1,), dtype=int, name=f"area_group"),
        tf.keras.Input(shape=(1,), dtype=str, name=f"object_type"),
        tf.keras.Input(shape=(1,), dtype=str, name=f"floor_group"),
        tf.keras.Input(shape=(1,), dtype=str, name=f"year_group"),
        tf.keras.Input(shape=(1,), dtype=str, name=f"street"),
        tf.keras.Input(shape=(1,), dtype=int, name=f"gvs"),
        tf.keras.Input(
            shape=(config["input_sequence_length"], config["n_features"]),
            name="LSTM input",
        ),
    ]
    layers = []
    for i, layer in enumerate(preprocessing_layers):
        layers.append(layer(inputs[i]))
    layers.append(inputs[-2])

    stat_features = tf.keras.layers.Concatenate(axis=-1, name="stat_features")(layers)
    X_stat = tf.keras.layers.Dense(
        config["stat_units_max"], activation="relu", name="dense_1"
    )(stat_features)
    X_stat = tf.keras.layers.Dense(
        config["stat_units_min"], activation="relu", name="dense_2"
    )(X_stat)

    encoder_output1, state_h1, state_c1 = tf.keras.layers.LSTM(
        config["n_units_max"],
        return_sequences=True,
        return_state=True,
        name="encoder_output1",
    )(inputs[-1])
    encoder_states1 = [state_h1, state_c1]

    encoder_output2, state_h2, state_c2 = tf.keras.layers.LSTM(
        config["n_units_min"], return_state=True, name="encoder_output2"
    )(encoder_output1)
    encoder_states2 = [state_h2, state_c2]

    decoder_inputs = tf.keras.layers.RepeatVector(
        config["output_sequence_length"], name="decoder_inputs"
    )(encoder_output2)
    decoder_l1 = tf.keras.layers.LSTM(
        config["n_units_max"], return_sequences=True, name="decoder_l1"
    )(decoder_inputs, initial_state=encoder_states1)

    repeat_stat = tf.keras.layers.RepeatVector(
        config["output_sequence_length"], name="repeat_stat"
    )(X_stat)
    concatenated = tf.keras.layers.Concatenate(axis=-1, name="all_features")(
        [decoder_l1, repeat_stat]
    )
    decoder_outputs2 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(config["n_features"]), name="time_distributed"
    )(concatenated)

    model = tf.keras.Model(inputs, decoder_outputs2)

    return model

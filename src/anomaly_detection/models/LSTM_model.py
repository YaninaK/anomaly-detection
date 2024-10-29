import logging
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ["define_model"]

MODEL_TYPE = "autoencoder"


def get_model(
    df_stat: pd.DataFrame,
    model_type: Optional[str] = None,
    config: Optional[dict] = None,
) -> tf.keras.Model:
    """
    Генерирует модель, работающую с последовательностями (LSTM) и
    статическими признаками.

    """
    if model_type is None:
        model_type = MODEL_TYPE

    if model_type == "autoencoder":
        from . import AUTOENCODER_CONFIG as CONFIG
    elif model_type == "ethalon_model":
        from . import ETHALON_MODEL_CONFIG as CONFIG

    if config is None:
        config = CONFIG

    norm_n_floors = tf.keras.layers.Normalization(axis=None, name="n_floors_prep")
    norm_n_floors.adapt(df_stat[config["features"][0]].fillna(0).values)

    norma_area = tf.keras.layers.Normalization(axis=None, name="area_prep")
    norma_area.adapt(df_stat[config["features"][1]].fillna(0).values)

    intlookup_area_group = tf.keras.layers.IntegerLookup(
        output_mode="one_hot", name=f"area_group_prep"
    )
    intlookup_area_group.adapt(df_stat[config["features"][2]])

    strlookup_object_type = tf.keras.layers.StringLookup(
        output_mode="int", name="object_type_prep"
    )
    strlookup_object_type.adapt(df_stat[config["features"][3]])
    n_vocab_1 = strlookup_object_type.vocabulary_size()
    n_emb_1 = np.ceil(np.sqrt(n_vocab_1)).astype(int)
    emb_object_type = tf.keras.layers.Embedding(
        n_vocab_1, n_emb_1, name="object_type_emb"
    )

    strlookup_floor_group = tf.keras.layers.StringLookup(
        output_mode="one_hot", name="floor_group_prep"
    )
    strlookup_floor_group.adapt(df_stat[config["features"][4]])

    strlookup_year_group = tf.keras.layers.StringLookup(
        output_mode="one_hot", name="year_group_prep"
    )
    strlookup_year_group.adapt(df_stat[config["features"][5]])

    strlookup_street = tf.keras.layers.StringLookup(
        output_mode="int", name="street_prep"
    )
    strlookup_street.adapt(df_stat[config["features"][6]])
    n_vocab_2 = strlookup_street.vocabulary_size()
    n_emb_2 = np.ceil(np.sqrt(n_vocab_2)).astype(int)
    emb_street = tf.keras.layers.Embedding(n_vocab_2, n_emb_2, name="street_emb")

    inputs = {
        "n_floors": tf.keras.Input(shape=(1,), dtype=int, name=f"n_floors"),
        "area": tf.keras.Input(shape=(1,), dtype=float, name=f"area"),
        "area_group": tf.keras.Input(shape=(1,), dtype=int, name=f"area_group"),
        "object_type": tf.keras.Input(shape=(1,), dtype=str, name=f"object_type"),
        "floor_group": tf.keras.Input(shape=(1,), dtype=str, name=f"floor_group"),
        "year_group": tf.keras.Input(shape=(1,), dtype=str, name=f"year_group"),
        "street": tf.keras.Input(shape=(1,), dtype=str, name=f"street"),
        "gvs": tf.keras.Input(shape=(1,), dtype=int, name=f"gvs"),
        "LSTM input": tf.keras.Input(
            shape=(config["input_sequence_length"], config["n_features"]),
            name="LSTM input",
        ),
    }
    layers = []
    layers.append(norm_n_floors(inputs["n_floors"]))
    layers.append(norma_area(inputs["area"]))
    layers.append(intlookup_area_group(inputs["area_group"]))

    object_type_strlookup = strlookup_object_type(inputs["object_type"])
    object_type_emb = emb_object_type(object_type_strlookup)
    object_type_dense = tf.keras.layers.Dense(
        config["object_type_units"], name="object_type_dense"
    )(object_type_emb)
    object_type_flatten = tf.keras.layers.Flatten(name="object_type_flatten")(
        object_type_dense
    )
    layers.append(object_type_flatten)

    layers.append(strlookup_floor_group(inputs["floor_group"]))
    layers.append(strlookup_year_group(inputs["year_group"]))

    street_strlookup = strlookup_street(inputs["street"])
    street_emb = emb_street(street_strlookup)
    street_dense = tf.keras.layers.Dense(config["street_units"], name="street_dense")(
        street_emb
    )
    street_flatten = tf.keras.layers.Flatten(name="street_flatten")(street_dense)
    layers.append(street_flatten)

    layers.append(inputs["gvs"])

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
    )(inputs["LSTM input"])
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

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from . import AUTOENCODER_CONFIG

logger = logging.getLogger(__name__)

__all__ = ["tensorflow_dataset_generation"]


def get_tf_dataset(df: pd.DataFrame, config: Optional[dict] = None) -> tf.data.Dataset:
    """
    Создает tensorflow dataset для обучения модели.
    """
    if config is None:
        config = AUTOENCODER_CONFIG

    ds = tf.data.Dataset.from_generator(
        lambda: generator(df, config["features"]),
        output_signature={
            "n_floors": tf.TensorSpec(shape=(1,), dtype=tf.int64, name="n_floors"),
            "area": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="area"),
            "area_group": tf.TensorSpec(shape=(1,), dtype=tf.int64, name="area_group"),
            "object_type": tf.TensorSpec(
                shape=(1,), dtype=tf.string, name="object_type"
            ),
            "floor_group": tf.TensorSpec(
                shape=(1,), dtype=tf.string, name="floor_group"
            ),
            "year_group": tf.TensorSpec(shape=(1,), dtype=tf.string, name="year_group"),
            "street": tf.TensorSpec(shape=(1,), dtype=tf.string, name="street"),
            "gvs": tf.TensorSpec(shape=(1,), dtype=tf.int64, name="gvs"),
            "LSTM input": tf.TensorSpec(
                shape=(4, 3), dtype=tf.float64, name="LSTM input"
            ),
        },
    )
    return ds


def generator(df: pd.DataFrame, features: list[str]) -> dict:
    """
    Генерирует данные для обучения модели.
    """
    df = df.sample(frac=1)
    for i in df.index:
        stat_inp = df.loc[i, features].values
        stat_inp = np.expand_dims(stat_inp, axis=1)
        inputs = {
            "n_floors": stat_inp[0].astype(int),
            "area": stat_inp[1].astype(float),
            "area_group": stat_inp[2].astype(int),
            "object_type": stat_inp[3],
            "floor_group": stat_inp[4],
            "year_group": stat_inp[5],
            "street": stat_inp[6],
            "gvs": stat_inp[7].astype(int),
            "LSTM input": np.array(df.loc[i, "LSTM input"]),
        }

        yield inputs

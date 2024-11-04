import logging
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from models import AUTOENCODER_CONFIG as CONFIG

logger = logging.getLogger(__name__)

__all__ = ["encode_static_features"]


def encode_stat_features(
    df_stat: pd.DataFrame, config: Optional[dict] = None
) -> np.array:
    """
    Готовит статические признаки для PCA-трансформации.
    Кодирует группу этажность объекта, группу год постройки через
    one_hot_encoding и добаляет к ним признак "Вид энерг-а ГВС"

    """
    if config is None:
        config = CONFIG

    strlookup_floor_group = tf.keras.layers.StringLookup(
        output_mode="one_hot", name="floor_group_prep"
    )
    strlookup_floor_group.adapt(df_stat[config["features"][4]])

    strlookup_year_group = tf.keras.layers.StringLookup(
        output_mode="one_hot", name="year_group_prep"
    )
    strlookup_year_group.adapt(df_stat[config["features"][5]])

    stat_features_list = [
        strlookup_floor_group(df_stat[config["features"][4]]).numpy(),
        strlookup_year_group(df_stat[config["features"][5]]).numpy(),
        np.expand_dims(df_stat[config["features"][-1]].values, 1),
    ]
    stat_features = np.hstack(stat_features_list)
    stat_features = stat_features[:, stat_features.sum(axis=0) > 0]

    return stat_features

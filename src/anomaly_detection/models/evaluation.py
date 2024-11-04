import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from . import AUTOENCODER_CONFIG as CONFIG

logger = logging.getLogger(__name__)

__all__ = ["evaluate_autoencoder_estimations"]


def evaluate_autoencoder_estimations(
    lstm_model: tf.keras.Model,
    ds_train: tf.data.Dataset,
    train_df: pd.DataFrame,
    batch_size: int,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Получает прогноз модели автоэнкодера, сопоставляет с фактическими значениями,
    расчитывает метрики и прикрепляет их к входящему датафрейму.
    """
    if config is None:
        config = CONFIG

    results = lstm_model.predict(ds_train.batch(batch_size))
    df = train_df["LSTM input"].apply(
        lambda x: pd.Series(
            np.array(x).T[0].tolist(),
            index=[f"true_{i}" for i in range(config["output_sequence_length"])],
        )
    )
    metrics = abs(df.values - results[:, :, 0]).mean(axis=1) / df.values.mean(axis=1)
    df.loc[:, [f"pred_{i}" for i in range(config["output_sequence_length"])]] = results[
        :, :, 0
    ]

    train_df = pd.concat([train_df, df], axis=1)
    train_df["Индекс соответствия прогнозу"] = metrics

    return train_df

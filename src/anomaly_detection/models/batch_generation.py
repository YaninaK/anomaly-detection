import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import AUTOENCODER_CONFIG

logger = logging.getLogger(__name__)

__all__ = ["batch_generation"]


def get_autoencoder_input_batch(
    batch_ind: list, df: pd.DataFrame, config: Optional[dict] = None
) -> Tuple[list[np.array], np.array]:
    """
    На основании списка из инедксов последовательностей, базы данных
    с последовательностями, генерирует пакет входных данных для автоэнкодера.
    """
    if config is None:
        config = AUTOENCODER_CONFIG

    stat_ind = []
    seq_list = []
    for ind in batch_ind:
        n = df[df["len_cumsum"] > ind].first_valid_index()
        i = ind + df.loc[n, "len_seq"] - df.loc[n, "len_cumsum"]

        stat_ind.append(n)

        seq = df["seq"][n][i]
        seq_list.append(seq)

    stat_inp = df.loc[stat_ind, config["features"]].values
    stat_inp = list(np.swapaxes(stat_inp, 0, 1))

    return stat_inp, np.array(seq_list)

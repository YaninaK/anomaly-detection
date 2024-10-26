import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["generate_model_inputs"]


SEQ_LENGTH = 4


def generate_model_inputs(
    df_seq: pd.DataFrame,
    temperature: pd.DataFrame,
    df_stat: pd.DataFrame,
    seq_length: Optional[int] = None,
) -> pd.DataFrame:

    if seq_length is None:
        seq_length = SEQ_LENGTH

    df = df_stat.copy()

    df["seq"] = df.reset_index()["index"].apply(
        generate_sequence_list,
        df=df_seq,
        temperature=temperature,
        seq_length=seq_length,
    )
    df["len_seq"] = df["seq"].apply(lambda x: len(x))
    df["len_cumsum"] = df["len_seq"].cumsum()

    return df


def generate_sequence_list(
    n: int,
    df: pd.DataFrame,
    temperature: pd.DataFrame,
    seq_length: int,
) -> list:
    """
    1. Совмещает данные о потреблении теплоэнергии с данными о температуре.
    2. Возвращает список последовательностей размерностью seq_length для объекта n.
    """
    x = df.iloc[n, :]
    sequence_list = []
    s = 0
    for ind, num in enumerate(x):
        if np.isnan(num):
            s = 0
        else:
            s += 1
            if s == seq_length:
                seq = [x[ind - k] for k in reversed(range(seq_length))]
                temp = [
                    temperature["t_scaled"][ind - k]
                    for k in reversed(range(seq_length))
                ]
                ozp = [temperature["ОЗП"][ind - k] for k in reversed(range(seq_length))]

                sequence_list.append(list(zip(seq, temp, ozp)))
                s = seq_length - 1

    return sequence_list

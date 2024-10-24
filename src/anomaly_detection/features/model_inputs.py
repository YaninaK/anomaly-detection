import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ["generate_model_inputs"]


SEQ_LENGTH = 4
FEATURES = [
    "Адрес объекта 2",
    "№ ОДПУ",
    "Тип объекта",
    "Вид энерг-а ГВС",
    "Группа этажность объекта",
    "Группа год постройки",
    "Группа общая площадь объекта",
]


def generate_model_inputs(
    df_seq: pd.DataFrame,
    temperature: pd.DataFrame,
    df_stat: pd.DataFrame,
    seq_length: Optional[int] = None,
    features: Optional[list] = None,
) -> list[Tuple[list, list, list]]:
    """
    Генерирует входные данные для моделей, работающих с последовательностями и
    статическими признаками.
    Для каждого объекта выводит:
    - индекс: ["Адрес объекта 2", "№ ОДПУ"]
    - статические признаки
    - последовательность длиною seq_length.
    """
    if seq_length is None:
        seq_length = SEQ_LENGTH
    if features is None:
        features = FEATURES

    input_list = []
    for i in tqdm(df_stat.index):
        sequence_list = generate_sequence_list(i, df_seq, temperature, seq_length)
        if sequence_list:
            stat_features = df_stat.loc[i, features].tolist()
            for sequence in sequence_list:
                input_list.append((stat_features[:2], stat_features[2:], sequence))

    return input_list


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
    t_ = temperature["t_scaled"]
    sequence_list = []
    s = 0
    for ind, num in enumerate(x):
        if np.isnan(num):
            s = 0
        else:
            s += 1
            if s == seq_length:
                seq = [x[ind - k] for k in reversed(range(seq_length))]
                temp = [t_[ind - k] for k in reversed(range(seq_length))]

                sequence_list.append(list(zip(seq, temp)))
                s = seq_length - 1

    return sequence_list

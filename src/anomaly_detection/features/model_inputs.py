import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "..", "src", "anomaly_detection"))


import logging
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ["generate_model_inputs"]


MODEL_TYPE = "autoencoder"
PATH = ""
FOLDER = "data/03_primary/"
FILE_NAME = "model_inputs_df.parquet.gzip"


class Generator:
    def __init__(
        self,
        model_type: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        if model_type is None:
            model_type = MODEL_TYPE

        if model_type == "autoencoder":
            from models import AUTOENCODER_CONFIG as CONFIG
        elif model_type == "ethalon_model":
            from models import ETHALON_MODEL_CONFIG as CONFIG

        if config is None:
            config = CONFIG

        self.config = config
        self.seq_length = config["seq_length"]
        self.file_name = None

    def fit_transform(
        self,
        df_seq: pd.DataFrame,
        temperature: pd.DataFrame,
        df_stat: pd.DataFrame,
        path: Optional[str] = None,
        folder: Optional[str] = None,
        file_name: str = None,
    ) -> tf.data.Dataset:
        """
        Создает tensorflow dataset для обучения модели по данным о потреблении теплоэнергии,
        статическим признакам, данным о температуре и количестве дней в отопительном периоде.
        """
        if path is None:
            path = PATH
        if folder is None:
            folder = f"{path}{FOLDER}"
        if file_name is None:
            file_name = FILE_NAME

        file_name = f"{folder}{file_name}"

        df_stat["Общая площадь объекта"].fillna(1, inplace=True)
        df_stat["Группа общая площадь объекта"].fillna(1, inplace=True)

        df = self.generate_model_inputs_df(df_seq, temperature, df_stat, file_name)
        dataset = self.get_tf_dataset(df)

        return dataset, df

    def get_tf_dataset(self, df: pd.DataFrame) -> tf.data.Dataset:
        """
        Создает tensorflow dataset на основе генератора данных.
        """
        ds = tf.data.Dataset.from_generator(
            lambda: self.generator(df),
            output_signature=(
                {
                    "n_floors": tf.TensorSpec(
                        shape=(1,), dtype=tf.int64, name="n_floors"
                    ),
                    "area": tf.TensorSpec(shape=(1,), dtype=tf.float64, name="area"),
                    "area_group": tf.TensorSpec(
                        shape=(1,), dtype=tf.int64, name="area_group"
                    ),
                    "object_type": tf.TensorSpec(
                        shape=(1,), dtype=tf.string, name="object_type"
                    ),
                    "floor_group": tf.TensorSpec(
                        shape=(1,), dtype=tf.string, name="floor_group"
                    ),
                    "year_group": tf.TensorSpec(
                        shape=(1,), dtype=tf.string, name="year_group"
                    ),
                    "street": tf.TensorSpec(shape=(1,), dtype=tf.string, name="street"),
                    "gvs": tf.TensorSpec(shape=(1,), dtype=tf.int64, name="gvs"),
                    "LSTM input": tf.TensorSpec(
                        shape=(self.config["input_sequence_length"], 3),
                        dtype=tf.float64,
                        name="LSTM input",
                    ),
                },
                tf.TensorSpec(
                    shape=(self.config["output_sequence_length"], 3),
                    dtype=tf.float64,
                    name="LSTM output",
                ),
            ),
        )
        return ds

    def generator(self, df: pd.DataFrame) -> dict:
        """
        Генерирует данные для обучения модели.
        """
        features = self.config["features"]
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
                "LSTM input": np.array(df.loc[i, "LSTM input"])[
                    : self.config["input_sequence_length"], :
                ],
            }
            label = np.array(df.loc[i, "LSTM input"])[
                -self.config["output_sequence_length"] :, :
            ]

            yield inputs, label

    def generate_model_inputs_df(
        self,
        df_seq: pd.DataFrame,
        temperature: pd.DataFrame,
        df_stat: pd.DataFrame,
        file_name: str,
    ) -> pd.DataFrame:

        df = df_stat.reset_index().copy()

        df["seq"] = df["index"].apply(
            self.generate_sequence_list,
            df=df_seq,
            temperature=temperature,
        )
        df = (
            df.merge(
                df["seq"].apply(pd.Series).stack().reset_index(),
                how="right",
                left_on="index",
                right_on="level_0",
            )
            .drop(["seq", "level_0"], axis=1)
            .rename(
                columns={
                    "index": "object index",
                    0: "LSTM input",
                    "level_1": "seq index",
                }
            )
        )
        df.to_parquet(file_name, compression="gzip")

        return df

    def generate_sequence_list(
        self,
        n: int,
        df: pd.DataFrame,
        temperature: pd.DataFrame,
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
                if s == self.seq_length:
                    seq = [x[ind - k] for k in reversed(range(self.seq_length))]
                    temp = [
                        temperature["t_scaled"][ind - k]
                        for k in reversed(range(self.seq_length))
                    ]
                    ozp = [
                        temperature["ОЗП"][ind - k]
                        for k in reversed(range(self.seq_length))
                    ]

                    sequence_list.append(list(zip(seq, temp, ozp)))
                    s = self.seq_length - 1

        return sequence_list

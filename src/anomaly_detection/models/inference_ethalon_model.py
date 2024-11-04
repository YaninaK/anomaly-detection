import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from data.data_sequence import generate_data_sequence
from data.preprocess import Preprocess
from features.model_inputs import Generator
from features.objects_grouping import ObjectsGrouping
from features.static_sequence_split import \
    generate_static_and_sequence_datasets
from features.temperature import transform_temperature

from . import ETHALON_MODEL_CONFIG as CONFIG

logger = logging.getLogger(__name__)

__all__ = ["preprocess_data_for_inference"]


def inference_data_preparation_pipeline(
    data: pd.DataFrame,
    temperature: pd.DataFrame,
    buildings: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[tf.data.Dataset, pd.DataFrame]:

    if config is None:
        config = CONFIG

    logging.info("Prefiltering data...")

    preprocess = Preprocess()
    data, buildings, temperature = preprocess.fit_transform(
        data, buildings, temperature
    )

    logging.info("Generating static dataset and consumption time series...")

    df = generate_data_sequence(data)
    df_stat, df_seq = generate_static_and_sequence_datasets(
        df, buildings, config=config
    )
    df_seq /= temperature["Число дней"]

    logging.info("Transforming temperature...")

    temperature = transform_temperature(temperature)

    logging.info("Grouping static features...")

    grouping = ObjectsGrouping()
    df_stat = grouping.fit_transform(df_stat)

    logging.info("Generating tensorflow datasets for ethalon model inference ...")

    generator_et = Generator(model_type="ethalon_model")

    ds_et_all, model_inputs_df = generator_et.fit_transform(
        df_seq, temperature, df_stat, file_name="model_inputs_df.parquet.gzip"
    )
    return ds_et_all, model_inputs_df

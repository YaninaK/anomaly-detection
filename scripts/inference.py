#!/usr/bin/env python3
"""Inference os sequence model for Anomaly detection"""

import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "anomaly_detection"))

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.anomaly_detection.data.make_dataset import load_data
from src.anomaly_detection.models.inference_ethalon_model import (
    inference_data_preparation_pipeline,
)
from src.anomaly_detection.models.inference_results import (
    post_process_inference_results,
)
from src.anomaly_detection.models.serialize import load_model

logger = logging.getLogger()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-d",
        "--data_folder_path",
        required=False,
        default="data/01_raw/",
        help="consumption, objects and temperature datasets store folder path",
    )
    argparser.add_argument(
        "-m",
        "--ethalon_model",
        required=False,
        default="ethalon_model_v1",
        help="model name",
    )
    argparser.add_argument(
        "-p",
        "--percentile",
        required=False,
        default=95,
        help="selected anomalies percentile",
    )
    argparser.add_argument(
        "-b",
        "--batch_size",
        required=False,
        default=128,
        help="batch_size for model inference",
    )
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Reading data...")

    data, temperature, buildings = load_data(args.data_folder_path)

    logging.info("Preprocessing data for inference...")

    ds_et_all, model_inputs_df = inference_data_preparation_pipeline(
        data, temperature, buildings
    )

    logging.info("Detecting duplicated heating consumption records...")

    ethalon_model = load_model(args.ethalon_model)

    logging.info("Inference...")

    results_et_all = ethalon_model.predict(ds_et_all.batch(args.batch_size))

    logging.info("Inference...")

    result, model_inputs_df = post_process_inference_results(
        model_inputs_df, results_et_all, args.percentile
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e)
        sys.exit(1)

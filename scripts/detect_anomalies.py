#!/usr/bin/env python3
"""Inference demo for Anomaly detection"""

import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))

import argparse
import logging

import pandas as pd
import tensorflow as tf

from src.anomaly_detection.data.make_dataset import load_data
from src.anomaly_detection.features.duplicated import (
    equal_values_identification_pipeline,
)
from src.anomaly_detection.features.missing_records import (
    missing_data_and_nonunique_objects_detection_pipeline,
)
from src.anomaly_detection.features.period_anomalies import (
    anomaly_detection_pipeline,
    select_anomalies,
)

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

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Reading data...")

    data, temperature, buildings = load_data(args.data_folder_path)

    logging.info(
        "Detecting missing consumption records, uninvoiced objects, nonunique objects..."
    )

    missing_consumption_records, uninvoiced_objects, nonunique_objects = (
        missing_data_and_nonunique_objects_detection_pipeline(
            data, buildings, temperature
        )
    )
    logging.info("Detecting duplicated heating consumption records...")

    completely_duplicated, equal_values = equal_values_identification_pipeline(
        data, buildings, temperature, save=save, path=PATH
    )

    logging.info("Detecting period anomalies...")

    (period_results, all_periods_anomalies, all_periods_anomalies_pivot) = (
        anomaly_detection_pipeline(
            data, temperature, buildings, alpha=5, beta=95, path=PATH
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e)
        sys.exit(1)

import logging
import os

import joblib
import tensorflow as tf

logger = logging.getLogger()

__all__ = ["store", "load"]


def store(model, model_history, filenames: list[str], path: str = "default"):
    if path == "default":
        path = models_path()

    model_filepath = os.path.join(path, filenames[0] + ".keras")
    model_history_filepath = os.path.join(path, filenames[1] + ".joblib")

    logger.info(f"Saving model in {model_filepath}...")

    model.save(model_filepath)

    logger.info(f"Dumpung model history into {model_history_filepath}...")

    joblib.dump(model_history, model_history_filepath)


def load_model(filename: str, path: str = "default"):
    if path == "default":
        path = models_path()

    filepath = os.path.join(path, filename + ".keras")

    logger.info(f"Loading model from {filepath}...")

    return tf.keras.models.load_model(filepath)


def load_model_history(filename: str, path: str = "default"):
    if path == "default":
        path = models_path()

    filepath = os.path.join(path, filename + ".joblib")

    logger.info(f"Loading model history from {filepath}...")

    return joblib.load(filepath)


def models_path() -> str:
    script_path = os.path.abspath(__file__)
    script_dir_path = os.path.dirname(script_path)
    models_folder = os.path.join(script_dir_path, "..", "..", "..", "models")

    return models_folder

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["post_process_inference_results"]


PERCENTILE = 95

COLS = [
    "Адрес объекта 2",
    "Тип объекта",
    "№ ОДПУ",
    "Вид энерг-а ГВС",
    "Этажность объекта",
    "Дата постройки",
    "Общая площадь объекта",
    "Период потребления",
    "Фактическое суточное потребление",
    "Прогноз модели",
    "Отклонение от прогноза",
    "Индекс соответствия прогнозу",
]

PATH = ""
FOLDER = "results/3_sequence_anomalies/"
FILE_NAMES = ["sequence_anomalies.xlsx", "all_scaled_seq_anomalies.xlsx"]


def post_process_inference_results(
    model_inputs_df: pd.DataFrame,
    results_et_all: np.array,
    percentile: Optional[int] = None,
    cols: Optional[list[str]] = None,
    path: Optional[str] = None,
    folder: Optional[str] = None,
    file_names: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if percentile is None:
        percentile = PERCENTILE
    if cols is None:
        cols = COLS
    if path is None:
        path = PATH
    if folder is None:
        folder = FOLDER
    if file_names is None:
        file_names = FILE_NAMES

    file_names = [f"{path}{folder}{name}" for name in file_names]

    df = model_inputs_df["LSTM input"].apply(
        lambda x: pd.Series(np.array(x).T[0].tolist())
    )

    model_inputs_df["Фактическое суточное потребление"] = df.values[:, -1]
    model_inputs_df["Прогноз модели"] = results_et_all[:, :, 0].flatten()
    model_inputs_df["Отклонение от прогноза"] = (
        model_inputs_df["Фактическое суточное потребление"]
        - model_inputs_df["Прогноз модели"]
    )
    model_inputs_df["Индекс соответствия прогнозу"] = (
        abs(model_inputs_df["Отклонение от прогноза"])
        / model_inputs_df["Фактическое суточное потребление"]
    )
    model_inputs_df.rename(
        columns={"last seq month": "Период потребления"}, inplace=True
    )

    q = np.percentile(abs(model_inputs_df["Отклонение от прогноза"]), percentile)
    result = model_inputs_df[abs(model_inputs_df["Отклонение от прогноза"]) > q]

    model_inputs_df[cols].to_excel(
        file_names[0],
        index=False,
    )
    result[cols].to_excel(
        file_names[1],
        index=False,
    )

    return result[cols], model_inputs_df[cols]

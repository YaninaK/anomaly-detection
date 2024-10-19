import logging

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["generate_data_sequence"]


def generate_data_sequence(data: pd.DataFrame) -> pd.DataFrame:
    """
    Создает последовательность данных без пропусков.
    """
    combined_index = sorted(
        list(
            set(
                zip(
                    data["Адрес объекта"],
                    data["Тип объекта"],
                    data["№ ОДПУ"],
                    data["Адрес объекта 2"],
                )
            )
        )
    )

    merge_basis = ["Адрес объекта", "Тип объекта", "№ ОДПУ", "Адрес объекта 2"]
    df = pd.DataFrame(combined_index, columns=merge_basis).merge(
        data.groupby(merge_basis)["Вид энерг-а ГВС"].first(), how="left", on=merge_basis
    )
    periods = sorted(data["Период потребления"].unique().tolist())
    for period in periods:
        current_period = (
            data[data["Период потребления"] == period][
                merge_basis + ["Текущее потребление, Гкал"]
            ]
            .rename(columns={"Текущее потребление, Гкал": period})
            .groupby(merge_basis, as_index=False)[period]
            .sum()
        )
        df = df.merge(current_period, how="left", on=merge_basis)

    df.set_index(merge_basis + ["Вид энерг-а ГВС"], inplace=True)

    return df

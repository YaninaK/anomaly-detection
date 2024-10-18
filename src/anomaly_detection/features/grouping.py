import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["grouping_buildings"]


class Grouping:
    def __init__(self):
        self.floor_labels = [
            "1-2 этажа",
            "3-4 этажа",
            "5-9 этажей",
            "10-12 этажей",
            "13 и более этажей",
        ]
        self.year_labels = [
            "до 1958 г",
            "1959-1989 гг.",
            "1990-2000 гг.",
            "2001-2010 гг.",
            "2011-2024 гг.",
        ]

    def fit_transform(
        self, data: pd.DataFrame, buildings: pd.DataFrame
    ) -> pd.DataFrame:
        """
        1. Для типа объекта Многоквартирный дом создает группу этажность объекта
        и группу год постройки
        2. Добавляет признак Вид энерг-а ГВС и данные по потреблению теплоэнергии многоквартирных
        домов.
        3. Оставляет только объекты с данными по потреблению теплоэнергии.
        4. Удаляет объекты с неуказанной Общей площадью объекта.

        """
        df = buildings[buildings["Тип Объекта"] == "Многоквартирный дом"].copy()
        df["Группа этажность объекта"] = pd.cut(
            df["Этажность объекта"],
            bins=[1, 2, 4, 9, 12, 99],
            labels=self.floor_labels,
            include_lowest=True,
        )
        df["Улица"] = df["Адрес объекта"].apply(lambda x: x.split(",")[1])
        df = self.fillnan_construction_date(df)
        df["Группа год постройки"] = pd.cut(
            df["Дата постройки 2"],
            bins=[
                pd.to_datetime(t, format="%Y")
                for t in [1800, 1959, 1990, 2001, 2011, 2025]
            ],
            labels=self.year_labels,
        )
        df = self.add_data_info(data, df)
        ind_to_drop = df[
            (df["Общая площадь объекта"] < 1) | (df["Общая площадь объекта"].isnull())
        ].index
        df = df.drop(ind_to_drop).reset_index()

        return df

    def fillnan_construction_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Заполняет медианами пропуски в дате постройки, опираясь на улицу,
        этажность объекта и группу этажности объекта
        """
        df["Дата постройки 2"] = df.groupby(["Улица", "Этажность объекта"])[
            "Дата постройки"
        ].transform(lambda x: x.fillna(x.median()))
        df["Дата постройки 2"] = df.groupby(["Улица", "Группа этажность объекта"])[
            "Дата постройки 2"
        ].transform(lambda x: x.fillna(x.median()))
        df["Дата постройки 2"] = df.groupby(["Группа этажность объекта"])[
            "Дата постройки 2"
        ].transform(lambda x: x.fillna(x.median()))

        return df

    def add_data_info(self, data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет признак Вид энерг-а ГВС и данные по потреблению теплоэнергии многоквартирных
        домов.
        Оставляет только объекты с данными по потреблению теплоэнергии.

        """
        cond = data["Тип объекта"] == "Многоквартирный дом"
        df = df.merge(
            data[cond]
            .groupby("Адрес объекта 2", as_index=False)["Вид энерг-а ГВС"]
            .first(),
            how="right",
            left_on="Адрес объекта 2",
            right_on="Адрес объекта 2",
        )
        df = df.merge(
            data[cond]
            .pivot_table(
                index="Адрес объекта 2",
                columns="Период потребления",
                values="Текущее потребление, Гкал",
            )
            .replace(0, np.nan)
            .reset_index(),
            how="right",
            left_on="Адрес объекта 2",
            right_on="Адрес объекта 2",
        )
        return df

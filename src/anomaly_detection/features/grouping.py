import logging

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
        Для типа объекта Многоквартирный дом создает группу этажность объекта
        и группу год постройки

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
        df.merge(
            data[data["Тип объекта"] == "Многоквартирный дом"]
            .groupby("Адрес объекта 2", as_index=False)["Вид энерг-а ГВС"]
            .first(),
            how="right",
            left_on="Адрес объекта 2",
            right_on="Адрес объекта 2",
            inplace=True,
        )

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

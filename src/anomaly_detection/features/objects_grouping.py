import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["grouping_buildings"]


class ObjectsGrouping:
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
        self.area_split = [0, 1, 2000, 2800, 3400, 3800, 4300, 5900, 8900, 25000, 65000]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Для типа объекта Многоквартирный дом создает группу этажность объекта
        и группу год постройки
        2. Добавляет признак Вид энерг-а ГВС и данные по потреблению теплоэнергии многоквартирных
        домов.
        3. Оставляет только объекты с данными по потреблению теплоэнергии.
        4. Удаляет объекты с неуказанной Общей площадью объекта.

        """
        df = self.generate_floor_group(df)
        df = self.fillnan_construction_date(df)
        df["Группа год постройки"] = pd.cut(
            df["Дата постройки 2"],
            bins=[
                pd.to_datetime(t, format="%Y")
                for t in [1800, 1959, 1990, 2001, 2011, 2025]
            ],
            labels=self.year_labels,
        )
        df["Группа общая площадь объекта"] = pd.cut(
            df["Общая площадь объекта"],
            bins=self.area_split,
            include_lowest=True,
            labels=self.area_split[1:],
        )

        return df

    def generate_floor_group(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Заполняет нулевое значение этажности единицами.
        Создает создает группы этажности объектов.
        """
        df["Этажность объекта"] = np.where(
            df["Этажность объекта"] == 0, 1, df["Этажность объекта"]
        )
        df["Группа этажность объекта"] = pd.cut(
            df["Этажность объекта"],
            bins=[1, 2, 4, 9, 12, 99],
            labels=self.floor_labels,
            include_lowest=True,
        )
        return df

    def fillnan_construction_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Заполняет медианами пропуски в дате постройки, опираясь на улицу,
        этажность объекта и группу этажности объекта
        """
        df["Улица"] = df["Адрес объекта 2"].apply(
            lambda x: x.split(",")[1] if len(x.split(",")) != 1 else ""
        )
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

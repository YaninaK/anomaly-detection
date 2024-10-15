import logging
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["preprocess_data"]


class Preprocess:
    def fit_transform(
        self, data: pd.DataFrame, buildings: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data = self.adjust_address_data(data)
        buildings = self.clean_buildings(buildings)
        buildings = self.adjust_address_buildings(buildings)
        data = self.adjust_subobjects_data(data, buildings)

        return data, buildings

    def adjust_address_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        1. Объект г Уфа, ул. Рихарда Зорге, д.27 разделен на 2 по № ОДПУ для приведения в соответствие
           с базой данных объектов.
        2. Объект г Уфа, ул. Владивостокская, д.10 корп.3,4 в data должен попасть в
           г Уфа, ул. Владивостокская, д.10 - корп.3,4 избыточен.
        3. Объекты г Уфа, ул. Даута Юлтыя, д.12 следует объединить - деление на подъезды несистемно:
            - г Уфа, ул. Даута Юлтыя, д.12,
            - г Уфа, ул. Даута Юлтыя, д.12, Подъезд №20654
            - г Уфа, ул. Даута Юлтыя, д.12, Подъезд №20655
        4. Объект г Уфа, ул. Комсомольская, д.15.. не содержит данных - совмещен с объектом
           г Уфа, ул. Комсомольская, д.15.

        """
        data["Адрес объекта 2"] = data["Адрес объекта"]
        data.loc[data["№ ОДПУ"] == "00111382", "Адрес объекта 2"] = (
            "г Уфа, ул. Рихарда Зорге, д.27, № ОДПУ 00111382"
        )
        data.loc[data["№ ОДПУ"] == "00102590", "Адрес объекта 2"] = (
            "г Уфа, ул. Рихарда Зорге, д.27, № ОДПУ 00102590"
        )

        data.loc[
            data["Адрес объекта"] == "г Уфа, ул. Владивостокская, д.10 корп.3,4",
            "Адрес объекта 2",
        ] = "г Уфа, ул. Владивостокская, д.10"
        data.loc[
            data["Адрес объекта"] == "г Уфа, ул. Даута Юлтыя, д.12, Подъезд №20654",
            "Адрес объекта 2",
        ] = "г Уфа, ул. Даута Юлтыя, д.12"
        data.loc[
            data["Адрес объекта"] == "г Уфа, ул. Даута Юлтыя, д.12, Подъезд №20655",
            "Адрес объекта 2",
        ] = "г Уфа, ул. Даута Юлтыя, д.12"
        data.loc[
            data["Адрес объекта"] == "г Уфа, ул. Комсомольская, д.15..",
            "Адрес объекта 2",
        ] = "г Уфа, ул. Комсомольская, д.15"

        return data

    def clean_buildings(self, buildings: pd.DataFrame) -> pd.DataFrame:
        """
        1. Удаляет строковые значения признака Этажность объекта
        2. Удаляет строковые значения признака Дата постройки
        3. Переводит значения признака Дата постройки в формат datetime
        4. Заполняет пропуски признака Дата постройки максимальным значением
            в разрезе объектов, расположенных по данному адресу.
        """
        df = buildings.copy()
        df["str"] = [type(s) == str for s in df["Этажность объекта"]]
        ind_to_drop = df[df["str"]].index
        buildings.drop(ind_to_drop, inplace=True)
        buildings["Этажность объекта"] = buildings["Этажность объекта"].astype(int)

        df["str"] = [type(s) == str for s in df["Дата постройки"]]
        ind_to_drop = df[df["str"]].index
        buildings.drop(ind_to_drop, inplace=True)
        buildings["Дата постройки"] = [
            i for i in pd.to_datetime(buildings["Дата постройки"])
        ]

        buildings["Дата постройки"] = buildings.groupby("Адрес объекта")[
            "Дата постройки"
        ].transform(lambda x: x.max())
        return buildings

    def adjust_address_buildings(self, buildings: pd.DataFrame) -> pd.DataFrame:
        """
        1. В базе данных объектов buildings неуникальные адреса многоквартирных
           домов заменяются на уникальные
        2. Задается уникальное название для объектов с пометкой extra, по которым
           не выставляются счета
        3. Уточняются адреса объектов для привязки баз data и buildings
        """
        buildings["Адрес объекта 2"] = buildings["Адрес объекта"]
        # 1. Неуникальные адреса объектов многоквартирных домов
        buildings.loc[1583, "Адрес объекта 2"] = (
            "г Уфа, ул. Вологодская, д.13, Подобъект №984990"
        )
        buildings.loc[2397, "Адрес объекта 2"] = (
            "г Уфа, ул. Кирова, д.95, Подобъект №46590"
        )
        buildings.loc[4320, "Адрес объекта 2"] = (
            "г Уфа, ул. Революционная, д.88, Подобъект №46372"
        )
        buildings.loc[5727, "Адрес объекта 2"] = (
            "г Уфа, ул. Энтузиастов, д.6, Подобъект №984984"
        )

        # 2. Объекты, на которые не выставляются счета
        ## 2.1 Многоквартирные дома
        buildings.loc[1603, "Адрес объекта 2"] = "г Уфа, ул. Вологодская, д.20, extra"
        buildings.loc[2101, "Адрес объекта 2"] = (
            "г Уфа, ул. Интернациональная, д.113, extra"
        )
        buildings.loc[3765, "Адрес объекта 2"] = "г Уфа, ул. Нежинская, д.6, extra"
        ## 2.2 Другое строение
        buildings.loc[1230, "Адрес объекта 2"] = (
            "г Уфа, ул. Баязита Бикбая, д.26, extra"
        )
        buildings.loc[1595, "Адрес объекта 2"] = (
            "г Уфа, ул. Вологодская, д.15 корп.1, extra"
        )

        # 3. Уточнение адреса объектов для привязки баз data и buildings
        ## 3.1 Многоквартирные дома
        buildings.loc[1908, "Адрес объекта 2"] = (
            "г Уфа, ул. Добролетная, д.7 корп.2, Подъезд №1"
        )
        buildings.loc[3864, "Адрес объекта 2"] = (
            "г Уфа, ул. Орджоникидзе, д.19 корп.2, Подъезд №985766"
        )
        buildings.loc[5804, "Адрес объекта 2"] = (
            "г Уфа, ул. Юрия Гагарина, д.41 корп.3, Подъезд №1"
        )
        ## 3.2 Общежитие
        buildings.loc[5118, "Адрес объекта 2"] = (
            "г Уфа, ул. Транспортная, д.44, Подобъект №32710"
        )
        ## 3.3 Учебное заведение, комбинат, центр
        buildings.loc[4395, "Адрес объекта 2"] = (
            "г Уфа, ул. Рихарда Зорге, д.27, № ОДПУ 00111382"
        )
        buildings.loc[4396, "Адрес объекта 2"] = (
            "г Уфа, ул. Рихарда Зорге, д.27, № ОДПУ 00102590"
        )
        ## 3.4 Другое строение
        buildings.loc[1140, "Адрес объекта 2"] = (
            "г Уфа, ул. Ахметова, д.316, Подобъект №986805"
        )
        buildings.loc[5192, "Адрес объекта 2"] = (
            "г Уфа, ул. Ульяновых, д.65, Подобъект №30886"
        )

        return buildings

    def adjust_subobjects_data(
        self, data: pd.DataFrame, buildings: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Укрупняет адрес объекта в базе data, признак Адрес объекта 2, чтобы привести
        в соответствие с базой buildings, признак Адрес объекта 2.
        """
        cond1_a = data["Тип объекта"] == "Многоквартирный дом"
        cond1_b = buildings["Тип Объекта"] == "Многоквартирный дом"
        addr = sorted(
            list(
                set(data[cond1_a]["Адрес объекта 2"])
                - set(buildings[cond1_b]["Адрес объекта 2"])
            )
        )
        adjusted_address = [(i.split("№")[0].replace(", Подобъект ", "")) for i in addr]
        unreconciled = set(adjusted_address) - set(
            buildings[cond1_b]["Адрес объекта 2"]
        )
        df = pd.DataFrame({"addr": addr, "adjusted_address": adjusted_address})
        unique_addr = df.loc[~df["adjusted_address"].isin(unreconciled)]

        data.loc[data["Адрес объекта"].isin(unique_addr["addr"]), "Адрес объекта 2"] = [
            (i.split("№")[0].replace(", Подобъект ", ""))
            for i in data.loc[
                data["Адрес объекта"].isin(unique_addr["addr"]), "Адрес объекта"
            ]
        ]
        return data

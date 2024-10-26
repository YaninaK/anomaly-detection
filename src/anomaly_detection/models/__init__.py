AUTOENCODER_CONFIG = {
    "features": [
        "Этажность объекта",
        "Общая площадь объекта",
        "Группа общая площадь объекта",
        "Тип объекта",
        "Группа этажность объекта",
        "Группа год постройки",
        "Улица",
        "Вид энерг-а ГВС",
    ],
    "n_features": 3,
    "stat_units_max": 64,
    "stat_units_min": 32,
    "input_sequence_length": 4,
    "output_sequence_length": 4,
    "n_units_max": 32,
    "n_units_min": 16,
}

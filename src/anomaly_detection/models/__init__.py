AUTOENCODER_CONFIG = {
    "model_type": "autoencoder",
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
    "n_periods": 24,
    "seq_length": 4,
    "n_features": 3,
    "object_type_units": 2,
    "street_units": 8,
    "stat_units_max": 16,
    "stat_units_min": 8,
    "input_sequence_length": 4,
    "output_sequence_length": 4,
    "n_units_max": 16,
    "n_units_min": 8,
}

ETHALON_MODEL_CONFIG = {
    "model_type": "ethalon_model",
    "quantile": 0.7,
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
    "seq_length": 4,
    "n_features": 3,
    "object_type_units": 2,
    "street_units": 8,
    "stat_units_max": 16,
    "stat_units_min": 8,
    "input_sequence_length": 3,
    "output_sequence_length": 1,
    "n_units_max": 16,
    "n_units_min": 8,
}

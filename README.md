# ТурбоХакатон: Решения для электроэнергетики на базе искусственного интеллекта
Решение задач на реальных датасетах компании Группы «Интер РАО»

### Содержание

1. [Постановкa задачи](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/tree/master/data-Y/anomaly-detection#%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7-%D0%B0%D0%BD%D0%BE%D0%BC%D0%B0%D0%BB%D0%B8%D0%B9-%D0%B2-%D0%BD%D0%B0%D1%87%D0%B8%D1%81%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F%D1%85-%D0%B7%D0%B0-%D1%82%D0%B5%D0%BF%D0%BB%D0%BE%D0%B2%D1%83%D1%8E-%D1%8D%D0%BD%D0%B5%D1%80%D0%B3%D0%B8%D1%8E)
2. [Решение](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/tree/master/data-Y/anomaly-detection#%D1%80%D0%B5%D1%88%D0%B5%D0%BD%D0%B8%D0%B5)
    * Исследование данных
    * Обнаружение нулевых значения показаний за тепловую энергию в отопительный период (октябрь-апрель).
    * Равные значения показаний в течение нескольких расчетных периодов.
    * Снижение/рост показаний в отдельные месяцы по сравнению с показаниями за предыдущие периоды по данному объекту (с учётом фактической температуры наружного воздуха и количества отопительных дней в месяце).
    * Аномально низкое/высокое (отклонение более 25%) потребление объекта в конкретном месяце по сравнению с аналогичными объектами для многоквартирных домов по группам, обозначенным в условии задачи.
3. [Демонстрация интерфейса решения](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/tree/master/data-Y/anomaly-detection#%D0%B4%D0%B5%D0%BC%D0%BE%D0%BD%D1%81%D1%82%D1%80%D0%B0%D1%86%D0%B8%D1%8F-%D0%B8%D0%BD%D1%82%D0%B5%D1%80%D1%84%D0%B5%D0%B9%D1%81%D0%B0-%D1%80%D0%B5%D1%88%D0%B5%D0%BD%D0%B8%D1%8F)

## Анализ аномалий в начислениях за тепловую энергию

В процессе работы с аномальными начислениями потребителям тепловой энергии были выявлены случаи ошибочно внесённых показаний, а также случаи аномально низкого потребления тепловой энергии. Это привело к неправильным начислениям на общую сумму T млн рублей в пользу потребителя. 

Необходимо разработать и внедрить систему искусственного интеллекта (ИИ), которая будет анализировать данные о потреблении тепловой энергии и выявлять аномальные начисления. 

Система должна учитывать различные факторы, такие как показания приборов учёта, договорные нагрузки, погодные условия и другие параметры, которые могут повлиять на потребление тепловой энергии. 

Система ИИ должна автоматически обнаруживать аномалии в данных о потреблении и предоставлять информацию о них ответственным сотрудникам для дальнейшего анализа и принятия решений.

### Критерии оценки:

* F-Score (50%) 
* Законченность решения (10%) 
* Оригинальность и эффективность решения (10%) 
* Качество исполнения (10%) 
* Оценка работы с данными (10%) 
* Презентация решения (10%)


### Примеры аномалий

Виды аномалий по показаниям приборов учёта тепловой энергии, которые необходимо выявлять, кроме объектов с видом энергопотребления ГВС (централ):

1. Нулевые значения показаний за тепловую энергию в отопительный период (октябрь-апрель).

2. Равные значения показаний в течение нескольких расчетных периодов.

3. Снижение/рост показаний в отдельные месяцы по сравнению с показаниями за предыдущие периоды по данному объекту (с учётом фактической температуры наружного воздуха и количества отопительных дней в месяце);

4. Аномально низкое/высокое (отклонение более 25%) потребление объекта в конкретном месяце по сравнению с аналогичными объектами (только для типов объекта «Многоквартирный дом») по критериям:

    * год постройки (по группам до 1958 г., 1959-1989 гг., 1990-2000 гг., 2001-2010 гг., 2011-2024 гг.),
    * этажность (по группам 1-2 этажа, 3-4 этажа, 5-9 этажей,10-12 этажей, 13 и более этажей),
    * площадь (±10%),
    * наличие ГВС ИТП (горячей воды, учитываемой тем же прибором).



## Решение

### Исследование данных

* Данные учета тепловой энергии за 24 месяца - 75391 строка.
* Данные об объектах: тип строения, этажность, площадь, год постройки - 5873 объекта.
* Данные о температуре месяца и продолжительности отопительного периода с октября по апрель.

[Здесь](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/blob/master/data-Y/anomaly-detection/notebooks/EDA_and_data_preprocessing.ipynb) ссылка на анализ данных и обоснование препроцессинга.

### Обнаружение нулевых значения показаний за тепловую энергию в отопительный период (октябрь-апрель).

* Для обнаружения пропусков в данных по текущему потреблению по каждому объекту воссоздавалась вся числовая последовательность месяцев в отопительный период.
* Объекты без данных по текущему потреблению обнаруживаются при сопоставлении  данных о потреблении и данных об объектах. Аналогично, для поиска объектов у которых ведется учет потребления, но они отсутствуют в базе объектов.
* Неуникальные адреса объектов обнаруживаются при поиске дубликатов в базе данных объектов.

При запуске кода сопоставляются данные о потреблении и данные об объектах и в [папке](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/tree/master/data-Y/anomaly-detection/results/1_missing_records) ```results/1_missing_records/``` генерируются 3 файла: 

* ```missing_records.xlsx``` - пропуски в показаниях за тепловую энергию в отопительный период
* ```uninvoiced_objects.xlsx``` - объекты без данных по текущему потреблению
* ```nonunique_objects.xlsx``` - неуникальные адреса объектов

[Здесь](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/blob/master/data-Y/anomaly-detection/notebooks/01_Missing_consumption_records.ipynb?ref_type=heads) ссылка демонстрацию результатов запуска кода.


### Равные значения показаний в течение нескольких расчетных периодов.

Дублирующиеся входные данные  и равные значения показаний в течение нескольких расчетных периодов обнаруживаются при поиске дубликатов в полной базе данных по потреблению тепловой энергии из всех периодов.

При запуске кода идентифицируются и в [папке](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/tree/master/data-Y/anomaly-detection/results/2_equal_records) ```results/2_equal_records/``` записываются в файлы:

* ```completely_duplicated.xlsx``` - дублирующиеся входные данные,
* ```equal_values.xlsx``` - равные значения показаний в течение нескольких расчетных периодов.

[Здесь](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/blob/master/data-Y/anomaly-detection/notebooks/02_Duplicates.ipynb?ref_type=heads) ссылка демонстрацию результатов запуска кода.

###  Снижение/рост показаний в отдельные месяцы по сравнению с показаниями за предыдущие периоды по данному объекту (с учётом фактической температуры наружного воздуха и количества отопительных дней в месяце).

* Аномалии плохо аппроксимируются моделями, работающими с короткими последовательностями (LSTM). Для погашения шумов, лучше использовать архитектуру автоэнкодера, когда последовательность предсказывает сама себя. 
    
* Инференс автоэнкодера играет роль предварительной разметки. Для обучения эталонной модели отбираем только “чистые” данные с небольшим отклонением прогноза от факта - 70% последовательностей из 4 периодов, которые хорошо поддаются обобщению автоэнкодером.

* Эталонная модель (LSTM) на основе последовательности из 3 периодов генерирует четвертый период. Обученная на “чистых” данных, эталонная модель будет чувствительна к аномалиям. 

* Предлагается не выбор 0/1 нет аномалии/ есть аномалия, а шкала, где наиболее подозрительные объекты будут иметь бОльшую величину отклонения прогноза от факта. Можно выбрать перцентиль, выше которого информация будет обрабатываться специалистами - 95 или 99 в зависимости от наличия ресурсов.

* Есть 2 варианта подхода к обработке результата: принимать во внимание абсолютное или относительное отклонение от прогноза от факта. Выбран первый вариант, поскольку в этом случае мы сосредотачиваемся на более крупных суммах, чтобы затраты времени специалистов окупились. 

В результате запуска кода  в [папке](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/tree/master/data-Y/anomaly-detection/results/3_sequence_anomalies) ```results/3_sequence_anomalies/``` генерируются файлы:

* ```sequence_anomalies.xlsx``` - отбор заданного перцентиля объектов с самым большим абсолютным отклонением от прогноза
* ```all_scaled_seq_anomalies.xlsx``` - все объекты прошедшие через инференс эталонной модели, с проставленными индексами аномальности.

Здесь [ссылка](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/blob/master/data-Y/anomaly-detection/notebooks/03_Sequence_anomalies_model.ipynb?ref_type=heads) на демонстрацию формирования каскадной модели и [ссылка](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/blob/master/data-Y/anomaly-detection/notebooks/03_Sequence_anomalies_inference.ipynb?ref_type=heads) на демонстрацию инференса модели.

Здесь [ссылка](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/blob/master/data-Y/anomaly-detection/scripts/train_save_model.py) на код обучения и сохранения модели из одной точки  и [ссылка](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/blob/master/data-Y/anomaly-detection/notebooks/03_Test_modul_train_save_model.ipynb?ref_type=heads) на демонстрацию работы этого кода, а также [ссылка](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/blob/master/data-Y/anomaly-detection/scripts/inference.py) на код инференса каскадной модели из одной точки.

### Аномально низкое/высокое (отклонение более 25%) потребление объекта в конкретном месяце по сравнению с аналогичными объектами для многоквартирных домов по группам, обозначенным в условии задачи.

Отклонение на 25% от медианы группы - достаточно грубая оценка аномалий, особенно, если группа малочисленная или в ней наблюдается большая дисперсия. PCA-разложение и расчет Hotelling's T-squared и Q residuals позволяет отрафинировать результат. Можно выбрать комбинацию условий, например, пресечение условий отклонение на 25% от медианы группы и наличие аномальных Hotelling's T-squared или Q residuals.

* На основании PCA трансформации данных о группах и удельном потреблении тепловой энергии на кв.метр площади за период, каждому объекту присваиваются значения Hotelling's T-squared и Q residuals, по которым можно определить, было ли потребление аномальным. 
* Параллельно рассчитываются медианные значения удельного потребления тепловой энергии в разрезе групп. Отклонение от медианы по группе более 25% считается аномалией. 
* Объекты с аномальными Hotelling's T-squared или Q residuals и удельным потреблением, отклоняющемся от медианы по группе более чем на 25% считаются аномальными.

В результате запуска кода в [папке](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/tree/master/data-Y/anomaly-detection/results/4_period_anomalies) ```results/4_period_anomalies``` генерируются файлы:

* ```all_periods_anomalies_pivot.xlsx``` - сводная таблица аномалий внутри периода
* ```all_periods_anomalies.xlsx``` - объекты с указанием индикаторов аномалий: Hotelling's T-squared, Q residuals, 25% ниже медианы,25% ниже медианы с указанием периода. 
* и папка ```period_data``` - со всеми данными, размеченными индикаторами аномалий.

Артефакты моделей и история обучения каждой из моделей находится в [папке](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/tree/master/data-Y/anomaly-detection/models) ```models```:

* ```autoencoder_v2.keras```
* ```autoencoder_training_history_v2.joblib```
* ```ethalon_model_v1.keras```
* ```ethalon_model_training_history_v1.joblib```

[Здесь](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/blob/master/data-Y/anomaly-detection/notebooks/04_Apartment_buildings.ipynb) ссылка демонстрацию результатов запуска кода.

[Здесь](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/blob/master/data-Y/anomaly-detection/scripts/detect_anomalies.py) ссылка на код, генерирующий результаты 1, 2 и 4 блоков, запускающийся из одной точки.


## Демонстрация интерфейса решения

Демонстрация интерфейса решения выполнена в виде web-приложения. [Здесь](https://git.codenrock.com/turboxakaton-reseniya-dlya-elektroenergetiki-na-baze-iskusstvennogo-intellekta-1268/cnrprod1728554803-team-80314/zagruzka-resheniya-6207/-/tree/master/data-Y/interrao) ссылка на код.
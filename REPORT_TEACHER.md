# Комментарии По Замечаниям Преподавателя

## 1. Валидационная ошибка и число эпох

В проекте `validation error` считается после каждой эпохи и используется как основной критерий выбора лучшей модели.

Где смотреть:

- [validation_error_epochs_ru.png](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_revision/training_graphs/validation_error_epochs_ru.png)
- [validation_epoch_argument.csv](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_revision/report_data/validation_epoch_argument.csv)
- [training_history.csv](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_revision/sgd_bottle/bottle/training_history.csv)

Актуальный итог по улучшенной `bottle`-модели:

- `best_epoch = 11`
- `best_val_loss = 0.380891`

Это означает, что модель выбирается не по последней эпохе, а по минимальной ошибке на validation.

## 2. Почему threshold не главный аргумент

`Threshold` относится к постобработке вероятностной карты модели.

Поэтому в проекте он вынесен в отдельное приложение:

- [threshold_appendix.csv](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_revision/report_data/threshold_appendix.csv)

Основное качество обучения доказывается через:

- `validation error`
- лучшую эпоху
- сравнение оптимизаторов
- итоговые `Dice` и `IoU`

## 3. Сравнение оптимизаторов

Были проверены несколько оптимизаторов и параметров обучения:

- [optimizer_argument.csv](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_revision/report_data/optimizer_argument.csv)

По исследованию лучший результат среди протестированных конфигураций дал:

- `SGD, lr = 1e-2`

Сравнение:

- `SGD`: `best_validation_error = 0.331616`
- `AdamW (5e-4)`: `0.402278`
- `Adam (5e-4)`: `0.408325`
- `AdamW (1e-3)`: `0.426022`
- `RMSprop (1e-4)`: `0.542996`

Итог: выбор финальной конфигурации сделан не на глаз, а по сравнению нескольких вариантов.

## 4. Что означают основные числа

Подготовлен словарь показателей:

- [metric_glossary.csv](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_revision/report_data/metric_glossary.csv)

Ключевые показатели:

- `train_loss` — ошибка на обучающей выборке
- `validation_error` — ошибка на validation
- `best_epoch` — эпоха с минимальной validation error
- `Dice` — качество перекрытия масок
- `IoU` — более строгая метрика перекрытия
- `learning_rate` — текущий шаг обучения

## 5. Актуальный итог по bottle

Итоговые файлы:

- [evaluation_summary.csv](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_revision/sgd_bottle/bottle/evaluation_summary.csv)
- [split_info.json](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_revision/sgd_bottle/bottle/split_info.json)

Финальные числа:

- `Dice = 0.538667`
- `IoU = 0.42929`
- `Threshold = 0.5`
- `Best epoch = 11`
- `Best validation loss = 0.380891`

## 6. Визуальные результаты

Для защиты подготовлены:

- примеры сегментации в `results/teacher_revision/sgd_bottle/bottle/examples/`
- contact sheets в `results/final/defense/showcase_visuals/`
- русские графики `validation error` по всем категориям в `results/teacher_revision/all_validation_graphs/`

## 7. Короткий вывод

Проект переведён в воспроизводимую учебную форму:

- одна основная модель `U-Net`
- контроль качества через `validation error`
- обоснование лучшей эпохи
- сравнение оптимизаторов
- численные метрики и визуальные примеры
- единая showcase-задача в ClearML

# Комментарии По Замечаниям Преподавателя

## 1. Валидационная ошибка добавлена

В проекте теперь явно считается `validation error` после каждой эпохи.

Где смотреть:

- [training_history.csv](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_review/bottle/training_history.csv)
- [training_curve.png](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_review/bottle/training_curve.png)
- [validation_error_curve.png](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_review/bottle/validation_error_curve.png)

`validation error` в таблице и есть контрольная ошибка на валидации.

## 2. Остановка обучения при росте ошибки

Используются два механизма:

- `ReduceLROnPlateau`: если валидационная ошибка ухудшается, уменьшается `learning rate`
- `Early stopping`: если улучшения нет несколько эпох подряд, обучение останавливается

Это нужно, чтобы не дообучать модель в момент переобучения.

## 3. Почему одной BCE недостаточно

Для сегментации применена комбинированная функция потерь:

`Loss = 0.6 * BCEWithLogitsLoss + 0.4 * DiceLoss`

Причина:

- `BCE` считает попиксельную бинарную ошибку
- при дефектах положительных пикселей мало, поэтому фон доминирует
- `pos_weight` компенсирует дисбаланс классов
- `DiceLoss` помогает лучше выделять форму дефекта

Поэтому требование "BCE должна быть 0" некорректно: важнее, чтобы уменьшалась именно `validation error`, а метрики `Dice` и `IoU` росли.

## 4. Что улучшено в обучении

- оптимизатор заменён на `AdamW`
- добавлен `weight_decay`
- добавлен `gradient clipping`
- добавлен `scheduler`
- добавлен `dropout` в глубоких слоях `U-Net`
- лучшая модель сохраняется по минимуму `validation error`, а не по последней эпохе

## 5. Итог по эксперименту bottle

Актуальные результаты лежат здесь:

- [evaluation_summary.csv](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_review/bottle/evaluation_summary.csv)
- [prediction_table.csv](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_review/bottle/prediction_table.csv)

Ключевые числа:

- `Best epoch`: 14
- `Best validation loss`: 0.402278
- `Dice`: 0.47287
- `IoU`: 0.357095

## 6. Визуальные результаты

Для показа преподавателю подготовлены изображения:

- [example_01.png](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_review/bottle/examples/example_01.png)
- [example_02.png](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_review/bottle/examples/example_02.png)
- [example_03.png](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_review/bottle/examples/example_03.png)
- [example_04.png](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_review/bottle/examples/example_04.png)
- [example_05.png](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_review/bottle/examples/example_05.png)
- [example_06.png](/Users/valeevrustemrafailevic/Projects/conveyor-defect-detection/results/teacher_review/bottle/examples/example_06.png)

На каждом примере показаны:

1. исходное изображение
2. истинная маска
3. предсказанная маска
4. наложение маски на изображение

## 7. Короткий вывод для защиты

В проекте реализована сегментация дефектов на основе `U-Net`.  
Качество модели контролируется по `validation error`, обучение автоматически замедляется и останавливается при ухудшении валидации.  
Для сегментации выбрана комбинированная функция ошибки `BCE + Dice`, так как она лучше работает при несбалансированных масках дефектов.  
Итог подтверждается численными метриками и визуальными примерами сегментации.

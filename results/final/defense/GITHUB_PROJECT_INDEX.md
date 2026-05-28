# Индекс Проекта Для GitHub И Защиты

## Репозиторий

- GitHub: [https://github.com/rvaleev2016-eng/conveyor-defect-detection](https://github.com/rvaleev2016-eng/conveyor-defect-detection)

## Структура проекта
Структура проекта организована таким образом, чтобы обеспечить:
- простую навигацию
- воспроизводимость экспериментов
- удобство анализа результатов
- разделение обучающих и итоговых артефактов

## Что открыть в первую очередь

1. `README.md`  
   Краткое описание проекта, структура, команды запуска, объяснение метрик и validation error.

2. `results/final/defense/full_project_analysis_report.docx`  
   Полный разбор проекта для сдачи.

3. `results/final/defense/clearml_defense_guide.docx`  
   Краткий маршрут показа в ClearML.

4. `results/teacher_revision/training_graphs/validation_error_epochs_ru.png`  
   Основной график по bottle для объяснения лучшей эпохи.

5. `results/teacher_revision/all_validation_graphs/validation_graphs_summary.csv`  
   Сводка по лучшим эпохам и validation error для всех категорий и общего обучения.

## Основные папки

- `models/` — архитектура `U-Net`
- `scripts/` — обучение, оценка, визуализация, сборка витрины ClearML
- `src/` — датасет, функции обучения, визуализация
- `results/final/defense/` — материалы для защиты
- `results/teacher_revision/` — исправления по замечаниям преподавателя

## Что показывать преподавателю

- одну итоговую задачу `Defense Showcase (U-Net)` в ClearML;
- график `validation error` и объяснение выбора числа эпох;
- сравнение отдельного обучения и общего обучения;
- визуалы сегментации;
- итоговую сводную таблицу.

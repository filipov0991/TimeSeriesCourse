# Brisbane Water Quality — Time Series Analysis

Итоговое задание по дисциплине **«Анализ временных рядов»**

## Описание
Анализ и прогнозирование **температуры воды** реки Брисбен (30-мин интервалы).  
Горизонт прогнозирования: **H=48 шагов (24 часа)**.

## Структура репозитория
```
.
├── data/                               # Данные
│   ├── brisbane_water_quality.csv      # Исходный датасет
│   └── prepared_temperature_ts.csv    # Подготовленный ВР
├── notebooks/
│   └── brisbane_ts_analysis.ipynb     # Основной ноутбук (EDA + все модели)
├── src/
│   ├── pipeline.py                    # Пайплайн (CLI запуск)
│   └── eda.py                         # EDA-модуль
├── output/                            # Результаты, метрики, графики
├── reports/
│   └── README.md                      # Полный аналитический отчёт
├── requirements.txt
└── .gitignore
```

## Быстрый старт

```bash
# 1. Создать virtualenv и установить зависимости
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
pip install -r requirements.txt

# 2. Запустить Jupyter
jupyter notebook notebooks/brisbane_ts_analysis.ipynb

# 3. Или запустить пайплайн напрямую
python src/pipeline.py
```

## Модели
| Тип | Модели |
|---|---|
| Статистические | SeasonalNaive, Naive, AutoARIMA, AutoETS, AutoTheta, MSTL |
| ML | LightGBM, RandomForest, XGBoost |
| DL | NHITS, N-BEATS, PatchTST |
| Аномалии | IQR, Z-score, Isolation Forest |

## Отчёт
Полный аналитический отчёт: [`reports/README.md`](reports/README.md)

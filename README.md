[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — NHL Salary Prediction — Предсказание зарплат игроков NHL

**Студент:** Урясьева Ксения Александровна

**Группа:** 231


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Статус CP1](#статус-cp1)
7. [Отчёт](#отчёт)


## Описание задачи

**Задача:** Регрессия - предсказание годовой зарплаты (cap hit) хоккеиста NHL
на основе его игровой статистики и биографических характеристик.

**Датасет:** данные собраны самостоятельно с помощью парсинга трёх источников:

- NHL Stats API (api.nhle.com) - игровая статистика за сезоны 2021-22 -> 2025-26
- NHL Web API (api-web.nhle.com) - биография игроков (рост, вес, драфт)
- Spotrac (spotrac.com) - cap hit контракты (целевая переменная)

Итоговый датасет: 4748 строк x 90 колонок (скейтеры), 525 x 46 (вратари).
После очистки: 2906 скейтеров, 286 вратарей.

**Целевая метрика:** Целевая метрика: MAE (средняя абсолютная ошибка в USD).
Дополнительно: RMSE, R2, MAPE.


## Структура репозитория
```
.
├── data
│   ├── raw                     # Исходные файлы с парсера
│   ├── processed               # После очистки и feature engineering
│   └── features                # Train/val/test сплиты
├── models                      # Сохранённые модели (.joblib)
├── notebooks
│   ├── 01_eda.ipynb            # EDA: первичный и вторичный анализ данных
│   ├── 02_baseline.ipynb       # Baseline и эксперименты с моделями
│   └── 03_experiments.ipynb    # Эксперименты: гиперпараметры, ансамбли, SHAP
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Графики и визуализации
│   └── report.md               # Финальный отчёт
├── src
│   ├── parser.py               # Парсер NHL API + Spotrac (Selenium)
│   ├── preprocessing.py        # Очистка, feature engineering, сплиты
│   └── modeling.py             # Обучение, оценка, сохранение моделей
├── tests
│   └── test_pipeline.py        # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

### Локально
 
```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>
 
# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
 
# 3. Установить зависимости
pip install -r requirements.txt
 
# 4. Запустить парсинг данных (опционально — данные уже в data/raw/)
python src/parser.py
 
# 5. Предобработка
python src/preprocessing.py
 
# 6. Обучение моделей
python src/modeling.py
```

## Данные
 
| Файл | Описание | Строк | Колонок |
|---|---|---|---|
| `data/raw/nhl_skaters_raw.csv` | Сырые данные скейтеров | 4748 | 90 |
| `data/raw/nhl_goalies_raw.csv` | Сырые данные вратарей | 525 | 46 |
| `data/processed/skaters_processed.csv` | После очистки и feature engineering | 2906 | 96 |
| `data/processed/goalies_processed.csv` | После очистки и feature engineering | 286 | 51 |
| `data/features/skaters_train.csv` | Train скейтеры (2022–2024) | 1829 | 96 |
| `data/features/skaters_val.csv` | Val скейтеры (2025) | 601 | 96 |
| `data/features/skaters_test.csv` | Test скейтеры (2026) | 476 | 96 |
 
Сплит - temporal: train на сезонах 2022–2024, val на 2025, test на 2026.
Случайный сплит не используется - cap hit назначается до начала сезона,
случайный сплит создаёт утечку данных.
 
 
## Результаты
 
### Скейтеры (оценка на val, сезон 2025)
 
| Модель | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| LinearRegression (baseline) | 1.307M | 1.869M | 0.581 | 46.0% |
| Ridge | 1.307M | 1.861M | 0.585 | 45.1% |
| Random Forest | 1.229M | 1.768M | 0.626 | 36.1% |
| XGBoost | 1.167M | 1.666M | 0.667 | 35.7% |
| **LightGBM** | **1.095M** | **1.602M** | **0.693** | **34.3%** |
| LightGBM + SelectFromModel (39 фичей) | 1.101M | 1.617M | 0.687 | 33.9% |
| PCA (37 компонент) + Ridge | 1.432M | 2.016M | 0.513 | 45.6% |
 
### Вратари (оценка на val, сезон 2025)
 
| Модель | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| LinearRegression (baseline) | 1.748M | 2.438M | 0.188 | 53.3% |
| Ridge | 1.709M | 2.339M | 0.253 | 48.3% |
| **Random Forest** | **1.483M** | **2.001M** | **0.453** | **43.7%** |
| XGBoost | 1.487M | 1.974M | 0.468 | 45.0% |
| LightGBM | 1.613M | 2.160M | 0.362 | 48.1% |


## Статус CP1
 
### Обработка и подготовка данных
- [x] Самостоятельный парсинг данных (NHL API + Spotrac через Selenium)
- [x] Описание датасета: источники, объём, колонки
- [x] Полная очистка: пропуски, дубли, выбросы, типы данных
- [x] Feature engineering: `age`, `draft_value`, `points_per_60`, `gsaa_proxy`,
      `starter_ratio`, `is_undrafted`, `is_multi_team`, `size_index`, `log_cap_hit`
- [x] Визуализации: распределения, корреляции, scatter plots, мультиколлинеарность,
      анализ по сезонам, категориальные признаки
- [x] Temporal split (train/val/test) с обоснованием защиты от leakage
- [x] Выбор метрик: MAE (приоритет), RMSE, R2, MAPE с обоснованием


### Моделирование и эксперименты
- [x] Baseline: LinearRegression без feature engineering
- [x] 4 модели: Ridge, Random Forest, XGBoost, LightGBM
- [x] Таблица экспериментов с метриками на val
- [x] Уменьшение размерности: SelectFromModel (39/77 фичей) и PCA + Ridge
- [x] Feature importance для лучших моделей

### Качество кода и воспроизводимость
- [x] Чистая структура проекта
- [x] `requirements.txt` с версиями
- [x] `random_state = 42` везде
- [ ] Линтер (ruff) 

## Отчёт

Финальный отчёт (пока нет): [`report/report.md`](report/report.md)

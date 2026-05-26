[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — NHL Salary Prediction — Предсказание зарплат игроков NHL

**Студент:** Урясьева Ксения Александровна

**Группа:** 231


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты-cp2)
6. [Статус CP1](#статус-cp1)
7. [Статус CP2](#статус-cp2)
8. [Статус CP3](#статус-cp3)
9. [Отчёт](#отчёт)


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
├── api
│   ├── __init__.py
│   ├── main.py                 # FastAPI приложение
│   ├── schemas.py              # Pydantic схемы
│   └── predictor.py            # Загрузка моделей, SHAP, предсказания
├── static
│   └── index.html              # Веб-интерфейс (HTML/CSS/JS)
├── tests
│   ├── test_pipeline.py        # Тесты пайплайна
│   └── test_api.py             # Smoke-тесты FastAPI
├── requirements.txt
├── README.md
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .pre-commit-config.yaml
├── pyproject.toml
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
 
# 4. Запустить парсинг данных (опционально если хочется посмотреть на данные реальные)
# Оно работает на винде с Edge определенной версии, для этого лежит msedgedriver.exe. 
# Если версия Edge другая, необходимо скачать такой же файл exe для своей версии и закинуть в корень проекта
python src/parser.py
 
# 5. Предобработка
python src/preprocessing.py
 
# 6. Обучение моделей
python src/modeling.py
```

## Docker

```bash
docker-compose up --build
```

После запуска доступно:
- `http://localhost:8000` — веб-интерфейс
- `http://localhost:8000/docs` — Swagger документация API
- `http://localhost:8888` — Jupyter (токен: nhl)


## Makefile

```bash
make lint-init   # первый запуск pre-commit (создаёт среду для хуков)
make lint      # проверка линтером (без изменений)
make lint-fix  # автоматическое исправление + форматирование      
make preprocess  # запуск preprocessing.py
make train       # запуск modeling.py
make test        # запуск тестов
make docker-up   # запуск docker-compose
make docker-down # остановка контейнеров
make api         # запуск FastAPI локально
make static      # запуск статики локально
```

## pre-commit

Хук автоматически запускает ruff перед каждым `git commit`.

Установка:
```bash
pip install pre-commit
pre-commit install
```

При первом запуске создаётся изолированная среда для хуков
(может занять 1-2 минуты):
```bash
pre-commit run --all-files
```

После этого хук будет запускаться автоматически при каждом коммите.
Если ruff найдёт ошибки — коммит не пройдёт, файлы будут исправлены
автоматически, их нужно добавить в `git add` и закоммитить снова.

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
 
 
## Результаты CP2

### Скейтеры: лучшие оценки на test, сезон 2026

| Модель | Val MAE | Test MAE | MAPE | R² |
|---|---|---|---|---|
| LinearRegression (baseline) | 1.275M | — | 41.3% | 0.603 |
| Ridge | 1.269M | — | 40.6% | 0.611 |
| ElasticNet tuned | 1.242M | — | 40.6% | 0.632 |
| XGBoost tuned | 1.063M | 1.287M | 33.0% | 0.648 |
| Stacking | 1.070M | 1.274M | 32.3% | 0.654 |
| **LightGBM + SelectFromModel (46 фичей)** | **1.080M** | **1.247M** | **31.5%** | **0.667** |

### Вратари: кросс-валидация по сезонам, лучшие оценки

| Модель | CV MAE | MAPE | R² |
|---|---|---|---|
| **CatBoost** | **1.251M** | **43.7%** | **0.529** |
| XGBoost tuned | 1.255M | 44.0% | 0.516 |
| CatBoost tuned | 1.260M | 43.5% | 0.521 |


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
- [x] Линтер (ruff) 

## Статус CP2

### Данные (улучшения)
- [x] Постановка Б: сдвиг таргета (статистика N -> контракт N+1)
- [x] Лаговые признаки для вратарей 
- [x] Interaction features для позиций (`points_per_60_x_forward`, `hits_per_60_x_defense` и др.)
- [x] Кодирование `birthCountry` (7 групп: CAN, USA, RUS, SWE, FIN, CZE, OTHER)
- [x] `age_squared` и `age_x_points_per_60` - нелинейность возраста

### Моделирование и эксперименты
- [x] 6 базовых моделей: Ridge, ElasticNet, Random Forest, XGBoost, LightGBM, CatBoost
- [x] Optuna тюнинг всех моделей (50 trials постановка А, 30 для Б)
- [x] Stacking ансамбль (LightGBM + XGBoost + CatBoost + RF -> Ridge)
- [x] Уменьшение размерности: SelectFromModel и PCA + Ridge
- [x] Две постановки задачи со сравнением
- [x] Кросс-валидация по сезонам для вратарей
- [x] SHAP анализ: summary plot, waterfall plots, dependence plots
- [x] Финальная оценка на test-сете, val vs test сравнение
- [x] Error analysis по позициям и диапазонам зарплат
- [x] Обоснование финальной модели с trade-off анализом

### Качество кода и воспроизводимость
- [x] Docker + docker-compose (jupyter + весь пайплайн в контейнере)
- [x] Makefile: `make lint`, `make lint-fix`, `make preprocess`, `make train`, `make test`, `make docker-up`
- [x] 13 smoke-тестов (pytest): данные, сплиты, leakage, модель
- [x] pre-commit хук (ruff автоматически перед каждым коммитом)

## Статус CP3

### Деплой
- [x] FastAPI с эндпоинтами predict/skater, predict/goalie, predict/batch, players/search
- [x] Веб-интерфейс на HTML/CSS/JS: ввод вручную, поиск игрока, пакетное предсказание
- [x] SHAP объяснения для каждого предсказания
- [x] Похожие игроки по рыночной стоимости
- [x] Слайдеры "что если" (интерактивное изменение предсказания)
- [x] Бейджи переоценён/недооценён для реальных игроков
- [x] Docker: api + jupyter в одном docker-compose
- [x] 14 smoke-тестов для API (pytest)

### Отчёт
- [x] Полный отчёт в Markdown по всем 8 секциям

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)

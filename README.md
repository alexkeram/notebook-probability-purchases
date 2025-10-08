# Прогноз вероятности покупки в 90 дней

## Цель
Предсказать вероятность, что клиент совершит покупку в течение 90 дней. Основная метрика — **ROC AUC**.

## Данные
Ожидаются в папке `data/`:
- `apparel-purchases.csv` — покупки (client_id, quantity, price, category_ids, date, message_id)
- `apparel-messages.csv` — события рассылок (bulk_campaign_id, client_id, message_id, event, channel, date, created_at)
- `apparel-target_binary.csv` — целевой признак по клиенту (client_id, target)
- `full_campaign_daily_event.csv` — агрегаты «день×кампания» (count_* и nunique_* по событиям)
- `full_campaign_daily_event_channel.csv` — то же с разбивкой по каналам

> В проекте все признаки строятся **из истории до T₀**; целевой период — **[T₀, T_max]**.  
> Для этого набора: **T₀ = 2023-11-18**, **T_max = 2024-02-16**, горизонт = **90 дней**.

## Как запустить
1. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
2. Поместите исходные CSV в `data/`.
3. Откройте ноутбук `notebook-probability-purchases.ipynb` и выполните шаги 1–15 (очистка → фичи → обучение → тест).

## Как воспроизвести модель из артефактов
В репозитории сохраняются артефакты обучения:
- `models/final_model.joblib` — обученная модель (лучшая по ROC AUC на валидации)
- `models/feature_list.txt` — список фичей (порядок колонок)
- `models/cfg.json` — служебный конфиг (T₀/T_max/горизонт/кол-во фичей)

Пример применения:
```python
import joblib, pandas as pd
from pathlib import Path

model = joblib.load("models/final_model.joblib")
feat_names = Path("models/feature_list.txt").read_text(encoding="utf-8").splitlines()

# X — матрица признаков с теми же колонками, что в feature_list.txt
proba = model.predict_proba(X[feat_names])[:, 1]
```

## Признаки (группы)
- **RFM (9):** `p_recency_days`, `p_freq_days`, `p_monetary`, `p_avg_basket`, `p_items_total`, `p_days_since_first`, `p_span_days`, `p_median_gap_days`, `p_share_with_msg`.
- **Категории (≈100):** `cat_*_cnt`, `cat_*_share`, `cat_total_tags`, `n_unique_cats`, `avg_cat_chain_len`. *Иерархия категорий «плавающая», поэтому используется позиционно‑инвариантный мешок id; берутся Top‑K по частоте.*
- **Реакция на рассылки (≈20):** `m_send/open/click/purchase/...`, `*_rate`, `m_last_*_days`, `m_nuniq_campaigns`, `m_send_email`, `m_send_push`.
- **Контекст кампаний (8):** `ctx_log_send_{mean,max}`, `ctx_open_share_{mean,max}`, `ctx_click_per_open_{mean,max}`, `ctx_conv_per_send_{mean,max}`.  
  *Важно:* `nunique_*` из дневных агрегатов **не суммируются между днями** — сначала соединяются с контактами клиента за конкретный день×кампанию, затем агрегируются по клиенту (mean/max).

Полный список фичей — в `models/feature_list.txt`.

## Результаты
- **Лучшая модель:** `HistGradientBoostingClassifier`
- **VALID (ROC AUC):** 0.700
- **TEST:** ROC AUC **0.722**, PR AUC **0.062**, LogLoss **0.364**
- **Lift:** top‑1% **6.52×**, top‑5% **3.79×**, top‑10% **3.19×**

## Структура репозитория
```
.
├── data/                           # CSV-файлы исходных данных
├── models/
│   ├── final_model.joblib
│   ├── feature_list.txt
│   └── cfg.json
├── notebook-probability-purchases.ipynb  # ноутбук (шаги 1–15)
├── README.md
└── requirements.txt
```

## Лицензия
Учебный проект. Используйте на свой страх и риск.

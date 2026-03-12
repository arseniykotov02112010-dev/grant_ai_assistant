"""
evaluate_faithfulness.py (финальная версия с прямыми HTTP-запросами)

Оценка faithfulness через YandexGPT API (нативный REST).
Использует requests, не зависит от openai-клиента.
"""

import os
import json
import re
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from tqdm import tqdm

# ==================== КОНФИГУРАЦИЯ ====================
YANDEX_CLOUD_FOLDER = "вставьте свое"
YANDEX_CLOUD_API_KEY = "вставьте свое"
YANDEX_CLOUD_MODEL = "yandexgpt/latest"    # или yandexgpt/rc

# Пути к файлам
BASE_DIR = Path(__file__).parent.parent
RESULTS_CSV = BASE_DIR / "evaluation" / "results_finetuned_rag" / "results_finetuned_rag.csv"
TEST_JSONL = BASE_DIR / "data" / "splits" / "test.jsonl"
OUTPUT_CSV = BASE_DIR / "evaluation" / "results_finetuned_rag" / "results_finetuned_rag_with_faithfulness.csv"
OUTPUT_REPORT = BASE_DIR / "evaluation" / "results_finetuned_rag" / "faithfulness_report.json"
LOG_FILE = BASE_DIR / "evaluation" / "logs" / "faithfulness_eval.log"

# Параметры запросов
TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_DELAY = 2            # секунд между попытками
REQUEST_DELAY = 1.0        # пауза между запросами (rate limit)
MAX_WORKERS = 1            # параллельные потоки (1 = безопасно)

# URL для YandexGPT (native completion)
YANDEX_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
MODEL_URI = f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}"

# ==================== ЛОГГИРОВАНИЕ ====================
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def clean_response(text: str) -> str:
    """Удаляет теги <think> и их содержимое."""
    if pd.isna(text):
        return ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'</?think>', '', cleaned)
    return cleaned.strip()

def load_test_contexts(test_jsonl: Path) -> Dict[Tuple[str, str], str]:
    contexts = {}
    duplicates = 0
    if not test_jsonl.exists():
        logger.error(f"Файл {test_jsonl} не найден!")
        return contexts

    with open(test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            key = (record['doc_id'], record['question'])
            if key in contexts:
                duplicates += 1
                logger.warning(f"Дубликат ключа (doc_id='{record['doc_id']}...'). Будет использован последний контекст.")
            contexts[key] = record.get('context', '').strip() or None

    logger.info(f"Загружено {len(contexts)} уникальных контекстов. Дубликатов: {duplicates}")
    return contexts

def load_results_data(results_csv: Path) -> pd.DataFrame:
    required_cols = ['doc_id', 'question', 'generated']
    if not results_csv.exists():
        logger.error(f"Файл {results_csv} не найден!")
        return pd.DataFrame()

    df = pd.read_csv(results_csv)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"В CSV отсутствуют обязательные колонки: {missing_cols}")
        return pd.DataFrame()

    logger.info(f"Загружено {len(df)} записей")
    return df

def create_prompt(context: str, question: str, answer: str) -> str:
    return f"""Ты — эксперт по оценке качества ответов ассистента. Тебе дан контекст документа, вопрос пользователя и ответ ассистента.

Определи, соответствует ли ответ ассистента фактам, изложенным в контексте. Игнорируй служебные теги вроде <think>, если они есть. Сосредоточься только на фактическом содержании.

Ответь только "ДА" или "НЕТ".

Контекст: {context}

Вопрос: {question}

Ответ ассистента: {answer}

Твой ответ (только "ДА" или "НЕТ"):"""

def evaluate_row(doc_id: str, question: str, generated: str, context: str) -> Optional[int]:
    """
    Оценивает faithfulness через прямой запрос к YandexGPT API.
    Возвращает 1 (ДА), 0 (НЕТ) или None в случае ошибки.
    """
    if pd.isna(generated):
        logger.warning(f"Пропуск: generated = NaN для (doc_id='{doc_id}')")
        return None

    cleaned_answer = clean_response(generated)
    if not cleaned_answer:
        logger.warning(f"Пустой ответ после очистки для (doc_id='{doc_id}')")
        return None

    prompt = create_prompt(context, question, cleaned_answer)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YANDEX_CLOUD_API_KEY}"
    }

    payload = {
        "modelUri": MODEL_URI,
        "messages": [
            {"role": "user", "text": prompt}
        ],
        "temperature": TEMPERATURE,
        "maxTokens": 10
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(YANDEX_API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                logger.error(f"HTTP {response.status_code}: {response.text}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                    continue
                else:
                    return None

            data = response.json()
            # Извлекаем текст ответа
            if not data.get("result") or not data["result"]["alternatives"]:
                logger.error("Неожиданный формат ответа API")
                return None

            answer_text = data["result"]["alternatives"][0]["message"]["text"].strip().upper()
            if answer_text.startswith('ДА'):
                return 1
            elif answer_text.startswith('НЕТ'):
                return 0
            else:
                logger.warning(f"Неожиданный ответ модели: '{answer_text}'. Попытка {attempt+1}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                    continue
                else:
                    return None

        except Exception as e:
            logger.error(f"Ошибка запроса (попытка {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                return None

    return None

def test_api_access() -> bool:
    """Проверяет доступ к API через простой запрос."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YANDEX_CLOUD_API_KEY}"
    }
    payload = {
        "modelUri": MODEL_URI,
        "messages": [
            {"role": "user", "text": "Скажи 'работает'"}
        ],
        "temperature": 0,
        "maxTokens": 10
    }
    try:
        response = requests.post(YANDEX_API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            answer = data["result"]["alternatives"][0]["message"]["text"].strip().lower()
            if answer == "работает":
                logger.info("✓ API-доступ проверен успешно")
                return True
            else:
                logger.warning(f"✓ API отвечает, но неожиданный ответ: {answer}")
                return True
        else:
            logger.error(f"Ошибка доступа к API: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.exception("Ошибка подключения к API. Проверьте ключ и folder_id.")
        return False

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def main():
    logger.info("=" * 60)
    logger.info("ЗАПУСК ОЦЕНКИ FAITHFULNESS")
    logger.info("=" * 60)

    contexts = load_test_contexts(TEST_JSONL)
    if not contexts:
        logger.error("Контексты не загружены. Выход.")
        return

    df = load_results_data(RESULTS_CSV)
    if df.empty:
        logger.error("Нет данных для обработки. Выход.")
        return

    if not test_api_access():
        return

    # Подготовка задач
    tasks = []
    for _, row in df.iterrows():
        doc_id = row['doc_id']
        question = row['question']
        generated = row['generated']
        context = contexts.get((doc_id, question))
        if context is None:
            logger.warning(f"Контекст не найден для (doc_id='{doc_id}', question='{question[:50]}...'). Пропускаем.")
            tasks.append((doc_id, question, generated, None))
        else:
            tasks.append((doc_id, question, generated, context))

    results = [None] * len(tasks)
    logger.info(f"Начало оценки {len(tasks)} примеров (потоков: {MAX_WORKERS})...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {}
        for idx, (doc_id, question, generated, context) in enumerate(tasks):
            if context is None:
                results[idx] = None
                continue
            future = executor.submit(evaluate_row, doc_id, question, generated, context)
            future_to_index[future] = idx

        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Оценка faithfulness"):
            idx = future_to_index[future]
            try:
                res = future.result()
                results[idx] = res
            except Exception as e:
                logger.error(f"Ошибка в потоке для индекса {idx}: {e}")
                results[idx] = None
            finally:
                time.sleep(REQUEST_DELAY)

    df['faithful'] = results

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    logger.info(f"Результаты сохранены в {OUTPUT_CSV}")

    # Подсчёт статистики
    total = len(df)
    evaluated_mask = df['faithful'].notna()
    evaluated_count = evaluated_mask.sum()
    faithful_count = df.loc[evaluated_mask, 'faithful'].sum()
    faithfulness_score = faithful_count / evaluated_count if evaluated_count > 0 else 0.0

    report = {
        "mode": "finetuned_rag_faithfulness",
        "total_examples_in_csv": total,
        "evaluated_examples": int(evaluated_count),
        "faithful_count": int(faithful_count),
        "unfaithful_count": int(evaluated_count - faithful_count),
        "faithfulness_score": round(faithfulness_score, 4),
        "notes": f"Faithfulness оценена через YandexGPT native API. Параллелизм: {MAX_WORKERS}, задержка: {REQUEST_DELAY}с."
    }

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Отчёт сохранён в {OUTPUT_REPORT}")

    logger.info("=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЁТ ПО FAITHFULNESS")
    logger.info("=" * 60)
    logger.info(f"Всего записей: {total}")
    logger.info(f"Оценено примеров: {evaluated_count}")
    logger.info(f"Faithfulness: {faithfulness_score:.2%}")
    logger.info(f"  - Соответствуют (ДА): {faithful_count}")
    logger.info(f"  - Не соответствуют (НЕТ): {evaluated_count - faithful_count}")
    logger.info(f"  - Не оценено: {total - evaluated_count}")
    logger.info("=" * 60)

if __name__ == "__main__":
    print("ПРЕДУПРЕЖДЕНИЕ: Перед запуском убедитесь, что вы:")
    print("1. Заменили YANDEX_CLOUD_FOLDER и YANDEX_CLOUD_API_KEY на СВОИ значения.")
    print("2. У вас есть средства на аккаунте Yandex Cloud.")
    print("3. Проверили пути к файлам.")
    main()
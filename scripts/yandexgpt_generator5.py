import json
import openai
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# ==================== КОНФИГУРАЦИЯ ====================
# ВАЖНО: Замените эти значения на СВОИ
YANDEX_CLOUD_FOLDER = "Ваш folder_id"
YANDEX_CLOUD_API_KEY = "Ваш IAM-токен"
YANDEX_CLOUD_MODEL = "yandexgpt/rc"  # Модель

# Настройка клиента OpenAI для YandexGPT
client = openai.OpenAI(
    api_key=YANDEX_CLOUD_API_KEY,
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    project=YANDEX_CLOUD_FOLDER
)

# Пути к файлам (адаптированные под вашу структуру)
BASE_DIR = Path(__file__).parent.parent
INPUT_DATA_PATH = BASE_DIR / "data" / "processed" / "documents_for_qa_generation.json"
OUTPUT_DATASET_PATH = BASE_DIR / "data" / "processed" / "sft_dataset.jsonl"
LOG_FILE_PATH = BASE_DIR / "data" / "processed" / "generation_log_fixed.txt"

# Создаем директории, если их нет
OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

# ==================== ЛОГГИРОВАНИЕ ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def estimate_tokens(text: str) -> int:
    """Приблизительная оценка количества токенов (1 токен ≈ 4 символа)."""
    return len(text) // 4


def split_document(content: str, max_tokens: int = 12000) -> List[Dict[str, Any]]:
    """
    Умное разделение документа по страницам.
    Возвращает список сегментов с метаданными.
    """
    segments = []
    pages = [p.strip() for p in content.split("=== Страница") if p.strip()]

    current_segment = []
    current_tokens = 0

    for i, page in enumerate(pages, start=1):
        page_tokens = estimate_tokens(page)

        if current_tokens + page_tokens > max_tokens and current_segment:
            segment_text = "\n\n".join(current_segment)
            segments.append({
                "text": segment_text,
                "page_range": f"{i - len(current_segment)}-{i - 1}",
                "token_estimate": current_tokens
            })
            current_segment = [page]
            current_tokens = page_tokens
        else:
            current_segment.append(page)
            current_tokens += page_tokens

    if current_segment:
        segment_text = "\n\n".join(current_segment)
        segments.append({
            "text": segment_text,
            "page_range": f"{len(pages) - len(current_segment) + 1}-{len(pages)}",
            "token_estimate": current_tokens
        })

    return segments


def generate_qa_pairs(document_chunk: str, doc_id: str, page_range: str, max_retries: int = 3) -> List[Dict[str, str]]:
    """
    Генерация QA-пар через YandexGPT API (OpenAI-совместимый протокол).
    """
    prompt = f"""Ты — эксперт по грантовой документации РНФ. На основе текста ниже создай 6-8 разных пар "вопрос-ответ".

ТРЕБОВАНИЯ К ВОПРОСАМ:
1. Разнообразные типы:
   - 2-3 фактологических (даты, суммы, цифры, сроки)
   - 2-3 интерпретационных (требования к участникам, условия, критерии)
   - 2 комплексных (перечислить этапы, объяснить процедуру)
2.Вопросы должны звучать естественно, как у реального исследователя.
3. Охватывать разные части текста.

ТРЕБОВАНИЯ К ОТВЕТАМ:
1. ТОЧНАЯ информация из текста. Никаких домыслов.
2. Если информация есть — прямая цитата или четкий пересказ.
3. Если информации нет — ответ "В данном фрагменте документа эта информация не указана."

ТЕКСТ ДОКУМЕНТА:
{document_chunk}

Верни В ТОЧНОМ ФОРМАТЕ JSON:
[{{"question": "...", "answer": "..."}}, ...]"""

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}",
                temperature=0.1,
                instructions="Ты создаешь качественные обучающие данные. Отвечаешь ТОЛЬКО в формате JSON.",
                input=prompt,
                max_output_tokens=4000
            )

            # Извлекаем и очищаем JSON
            answer_text = response.output_text.strip()
            if answer_text.startswith('```json'):
                answer_text = answer_text[7:-3].strip()
            elif answer_text.startswith('```'):
                answer_text = answer_text[3:-3].strip()

            qa_pairs = json.loads(answer_text)

            # Добавляем метаданные
            enhanced_pairs = []
            for pair in qa_pairs:
                enhanced_pairs.append({
                    "doc_id": doc_id,
                    "context_chunk_preview": document_chunk[:500] + "...",
                    "page_range": page_range,
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "metadata": {
                        "source": "yandexgpt-rc",
                        "generation_timestamp": time.time()
                    }
                })

            return enhanced_pairs

        except json.JSONDecodeError as e:
            logger.error(f"Попытка {attempt+1}/{max_retries}: Ошибка парсинга JSON для {doc_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return []
        except Exception as e:
            logger.error(f"Попытка {attempt+1}/{max_retries}: Ошибка API для {doc_id}: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Экспоненциальная задержка
            else:
                return []

    return []

def test_api_access():
    """Тестовый запрос для проверки доступности API."""
    try:
        test_response = client.responses.create(
            model=f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}",
            temperature=0.1,
            instructions="Ответь одним словом.",
            input="Скажи 'работает'",
            max_output_tokens=10
        )
        if test_response.output_text.strip().lower() == "работает":
            logger.info("✓ API-доступ проверен успешно")
            return True
        else:
            logger.warning(f"✓ API отвечает, но неожиданный ответ: {test_response.output_text}")
            return True
    except Exception as e:
        logger.error(f"✗ Ошибка подключения к API: {type(e).__name__}: {e}")
        return False

# ==================== ОСНОВНОЙ ПАЙПЛАЙН ====================

def main():
    """Основной пайплайн генерации датасета."""
    logger.info("=" * 60)
    logger.info("ЗАПУСК ГЕНЕРАЦИИ СИНТЕТИЧЕСКОГО ДАТАСЕТА")
    logger.info("=" * 60)

    # 1. Проверка API
    if not test_api_access():
        logger.error("Проверка аутентификации не пройдена. Генерация остановлена.")
        return

    # 2. Загрузка подготовленных документов
    if not INPUT_DATA_PATH.exists():
        logger.error(f"Входной файл не найден: {INPUT_DATA_PATH}")
        logger.error("Убедитесь, что вы выполнили скрипт prepare_for_generation.py")
        return

    with open(INPUT_DATA_PATH, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    logger.info(f"Загружено документов: {len(documents)}")

    # 3. Подготовка к генерации
    total_examples = 0
    processed_docs = 0
    estimated_cost_rub = 0.0
    PRICE_PER_1000_TOKENS = 15.0  # руб. за 1K токенов генерации

    # 4. Обработка документов
    with open(OUTPUT_DATASET_PATH, 'a', encoding='utf-8') as out_file:
        for doc in documents:
            doc_id = doc["doc_id"]
            logger.info(f"\nОбработка документа: {doc_id}")

            # Сегментация документа
            segments = split_document(doc["content"])
            logger.info(f"  Разбит на сегментов: {len(segments)}")

            for seg_num, segment in enumerate(segments, 1):
                logger.info(f"  Сегмент {seg_num}: стр. {segment['page_range']}, ~{segment['token_estimate']} токенов")

                # Генерация QA-пар
                qa_pairs = generate_qa_pairs(segment["text"], doc_id, segment["page_range"])

                if not qa_pairs:
                    logger.warning(f"    Не удалось сгенерировать QA-пары для сегмента")
                    continue

                # Сохранение результатов
                for qa in qa_pairs:
                    out_file.write(json.dumps(qa, ensure_ascii=False) + '\n')

                # Расчет стоимости (примерный)
                output_tokens = sum(estimate_tokens(qa["answer"]) for qa in qa_pairs)
                segment_cost = (output_tokens / 1000) * PRICE_PER_1000_TOKENS
                estimated_cost_rub += segment_cost
                total_examples += len(qa_pairs)

                logger.info(f"    Сгенерировано QA-пар: {len(qa_pairs)}")
                logger.info(f"    Накопленная стоимость: ~{estimated_cost_rub:.2f} руб.")
                logger.info(f"    Всего примеров: {total_examples}")

                # Пауза между запросами
                time.sleep(0.5)

            processed_docs += 1
            logger.info(f"Прогресс: {processed_docs}/{len(documents)} документов обработано")

    # 5. Финальный отчет
    logger.info("\n" + "=" * 60)
    logger.info("ГЕНЕРАЦИЯ ЗАВЕРШЕНА")
    logger.info("=" * 60)
    logger.info(f"Итоговая статистика:")
    logger.info(f"  Обработано документов: {processed_docs}/{len(documents)}")
    logger.info(f"  Сгенерировано QA-пар: {total_examples}")
    logger.info(f"  Примерная стоимость: ~{estimated_cost_rub:.2f} руб.")
    logger.info(f"  Данные сохранены в: {OUTPUT_DATASET_PATH}")
    logger.info(f"  Лог сохранен в: {LOG_FILE_PATH}")

if __name__ == "__main__":
    # Проверка установки библиотеки
    try:
        import openai
    except ImportError:
        print("Библиотека openai не установлена. Установите её:")
        print("pip install openai")
        exit(1)

    print("ПРЕДУПРЕЖДЕНИЕ: Перед запуском убедитесь, что вы:")
    print("1. Заменили YANDEX_CLOUD_FOLDER и YANDEX_CLOUD_API_KEY на СВОИ значения")
    print("2. Пополнили баланс в Яндекс.Облаке (рекомендуется 600+ руб.)")
    print("3. Активировали сервис YandexGPT")

    confirmation = input("\nВсе настроено и готово? (y/n): ")
    if confirmation.lower() == 'y':
        main()
    else:
        print("Запуск отменен. Настройте доступ к API.")

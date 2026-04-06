#!/usr/bin/env python3
"""
VK-бот для RAG-сервиса грантовых документов.
Версия с клавиатурой, прогресс-баром и примерами вопросов.
"""

import os
import logging
import time
import html
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.utils import get_random_id
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from dotenv import load_dotenv

# Загружаем переменные окружения
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Конфигурация
VK_TOKEN = os.getenv("VK_TOKEN")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", 3600))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 20))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", 1000))
MAX_HISTORY_STORED = int(os.getenv("MAX_HISTORY_STORED", 200))
MAX_HISTORY_SHOWN = int(os.getenv("MAX_HISTORY_SHOWN", 20))
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", 2))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 1))

# Состояния
WAITING_DOC, WAITING_QUESTION = 0, 1

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

user_data: Dict[int, Dict] = {}

# ==================== КЛАВИАТУРА ====================

def get_main_keyboard():
    keyboard = VkKeyboard(one_time=False)
    keyboard.add_button('📄 Новый диалог', color=VkKeyboardColor.PRIMARY)
    keyboard.add_button('📜 История', color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button('ℹ️ Информация', color=VkKeyboardColor.SECONDARY)
    keyboard.add_button('🧹 Очистить историю', color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button('❓ Помощь', color=VkKeyboardColor.PRIMARY)
    return keyboard

# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С БЭКЕНДОМ ====================

def check_backend_health() -> Tuple[bool, str]:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if resp.status_code == 200:
            return True, "OK"
        else:
            return False, f"HTTP {resp.status_code}"
    except requests.Timeout:
        return False, "timeout"
    except requests.ConnectionError:
        return False, "connection refused"
    except Exception as e:
        return False, str(e)


def backend_upload(file_bytes: bytes, filename: str, user_id: int) -> Tuple[Optional[str], Optional[int], str]:
    files = {'file': (filename, file_bytes)}
    try:
        resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
        if resp.status_code == 200:
            try:
                result = resp.json()
            except Exception as e:
                logger.error(f"User {user_id}: Failed to parse JSON from upload response: {e}")
                return None, 500, "Invalid JSON response from server"
            session_id = result.get("session_id")
            if isinstance(session_id, str):
                return session_id, 200, ""
            else:
                return None, 200, "Invalid response format (missing session_id)"
        else:
            logger.error(f"User {user_id}: Upload error {resp.status_code}: {resp.text[:200]}")
            return None, resp.status_code, f"HTTP {resp.status_code}"
    except requests.Timeout:
        logger.error(f"User {user_id}: Upload timeout")
        return None, 408, "Request timeout"
    except requests.ConnectionError as e:
        logger.error(f"User {user_id}: Cannot connect to backend: {e}")
        return None, 503, "Cannot connect to backend"
    except Exception as e:
        logger.exception(f"User {user_id}: Unexpected error in backend_upload")
        return None, 500, str(e)


def backend_ask(session_id: str, question: str, user_id: int, retry: int = 0) -> Tuple[Optional[str], Optional[int], str]:
    payload = {"session_id": session_id, "question": question}
    try:
        resp = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=120)
        if resp.status_code == 200:
            try:
                result = resp.json()
            except Exception as e:
                logger.error(f"User {user_id}: Failed to parse JSON from ask response: {e}")
                return None, 500, "Invalid JSON response from server"
            answer = result.get("answer")
            if isinstance(answer, str):
                return answer, 200, ""
            else:
                return None, 200, "Invalid response format (missing or non-string answer)"
        elif resp.status_code >= 500 and retry < MAX_RETRIES:
            logger.warning(f"User {user_id}: Server error {resp.status_code}, retry {retry+1}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY * (retry + 1))
            return backend_ask(session_id, question, user_id, retry + 1)
        else:
            logger.error(f"User {user_id}: Ask error {resp.status_code}: {resp.text[:200]}")
            return None, resp.status_code, f"HTTP {resp.status_code}"
    except requests.Timeout:
        logger.error(f"User {user_id}: Ask timeout")
        return None, 408, "Request timeout"
    except requests.ConnectionError as e:
        logger.error(f"User {user_id}: Cannot connect to backend: {e}")
        return None, 503, "Cannot connect to backend"
    except Exception as e:
        logger.exception(f"User {user_id}: Unexpected error in backend_ask")
        return None, 500, str(e)


# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С ПОЛЬЗОВАТЕЛЬСКИМИ ДАННЫМИ ====================

def get_user_data(user_id: int) -> Dict[str, Any]:
    if user_id not in user_data:
        user_data[user_id] = {
            "state": WAITING_DOC,
            "history": [],
            "session_id": None,
            "filename": None,
            "session_created": None,
            "last_question_time": 0
        }
    return user_data[user_id]


def add_to_history(user_data: Dict[str, Any], role: str, content: str):
    history = user_data.setdefault("history", [])
    history.append({"role": role, "content": content, "time": time.time()})
    if len(history) > MAX_HISTORY_STORED:
        user_data["history"] = history[-MAX_HISTORY_STORED:]


def get_history_text(user_data: Dict[str, Any]) -> str:
    history = user_data.get("history", [])
    if not history:
        return "📭 История пуста."
    entries = []
    for entry in history[-MAX_HISTORY_SHOWN:]:
        dt = datetime.fromtimestamp(entry["time"]).strftime("%H:%M")
        prefix = "👤" if entry["role"] == "user" else "🤖"
        entries.append(f"{prefix} [{dt}] {entry['content']}")
    return "\n\n".join(entries)


def clear_history(user_data: Dict[str, Any]):
    user_data["history"] = []


def get_session_info(user_data: Dict[str, Any]) -> str:
    filename = user_data.get("filename", "неизвестно")
    created = user_data.get("session_created")
    if created is None:
        return f"📄 Документ: {filename}\n⏱️ Время создания сессии неизвестно."
    elapsed = time.time() - created
    remaining = max(0, SESSION_TIMEOUT - elapsed)
    remaining_str = str(timedelta(seconds=int(remaining)))
    return f"📄 Документ: {filename}\n⏱️ Осталось времени сессии: {remaining_str}"


def check_rate_limit(user_data: Dict[str, Any]) -> bool:
    last = user_data.get("last_question_time", 0)
    if time.time() - last < RATE_LIMIT_SECONDS:
        return False
    user_data["last_question_time"] = time.time()
    return True


# ==================== ОТПРАВКА СООБЩЕНИЙ ====================

def send_message(vk, user_id, text, keyboard=None):
    safe_text = html.escape(text)
    params = {
        "user_id": user_id,
        "message": safe_text,
        "random_id": get_random_id()
    }
    if keyboard:
        params["keyboard"] = keyboard.get_keyboard()
    vk.messages.send(**params)


def send_typing(vk, user_id):
    vk.messages.setActivity(user_id=user_id, type="typing")


def handle_button(ud, text, user_id, vk):
    if text == '📄 Новый диалог':
        ud.clear()
        ud.update({
            "state": WAITING_DOC,
            "history": [],
            "session_id": None,
            "filename": None,
            "session_created": None,
            "last_question_time": 0
        })
        send_message(vk, user_id, "Начинаем новый диалог. Отправьте документ (PDF/DOCX).", get_main_keyboard())
        return True
    elif text == '📜 История':
        history_text = get_history_text(ud)
        send_message(vk, user_id, history_text, get_main_keyboard())
        return True
    elif text == 'ℹ️ Информация':
        info_text = get_session_info(ud)
        send_message(vk, user_id, info_text, get_main_keyboard())
        return True
    elif text == '🧹 Очистить историю':
        clear_history(ud)
        send_message(vk, user_id, "🧹 История очищена.", get_main_keyboard())
        return True
    elif text == '❓ Помощь':
        help_text = (
            "📚 **Справка**\n\n"
            "1. Отправьте мне PDF или DOCX файл (до 20 МБ).\n"
            "2. После загрузки задавайте вопросы по документу.\n"
            "3. Сессия живёт 1 час после последнего вопроса.\n\n"
            "**Кнопки:**\n"
            "📄 Новый диалог – начать заново\n"
            "📜 История – показать последние сообщения\n"
            "ℹ️ Информация – данные о текущем документе\n"
            "🧹 Очистить историю – удалить историю\n"
            "❓ Помощь – эта справка"
        )
        send_message(vk, user_id, help_text, get_main_keyboard())
        return True
    return False


def process_message(event, vk):
    user_id = event.user_id
    text = event.text.strip()
    ud = get_user_data(user_id)

    logger.info(f"process_message: user={user_id}, text={text}")

    # Обработка старых команд (для совместимости)
    if text.startswith('/'):
        if text == '/start':
            ud.clear()
            ud.update({
                "state": WAITING_DOC,
                "history": [],
                "session_id": None,
                "filename": None,
                "session_created": None,
                "last_question_time": 0
            })
            send_message(vk, user_id, "Привет! Отправьте мне PDF или DOCX файл.", get_main_keyboard())
            return
        elif text == '/help':
            help_text = (
                "📚 **Справка**\n\n"
                "1. Отправьте мне PDF или DOCX файл (до 20 МБ).\n"
                "2. После загрузки задавайте вопросы по документу.\n"
                "3. Сессия живёт 1 час после последнего вопроса.\n\n"
                "**Кнопки:**\n"
                "📄 Новый диалог – начать заново\n"
                "📜 История – показать последние сообщения\n"
                "ℹ️ Информация – данные о текущем документе\n"
                "🧹 Очистить историю – удалить историю\n"
                "❓ Помощь – эта справка"
            )
            send_message(vk, user_id, help_text, get_main_keyboard())
            return
        elif text == '/new':
            ud.clear()
            ud.update({
                "state": WAITING_DOC,
                "history": [],
                "session_id": None,
                "filename": None,
                "session_created": None,
                "last_question_time": 0
            })
            send_message(vk, user_id, "Начинаем новый диалог. Отправьте документ (PDF/DOCX).", get_main_keyboard())
            return
        elif text == '/history':
            history_text = get_history_text(ud)
            send_message(vk, user_id, history_text, get_main_keyboard())
            return
        elif text == '/clear':
            clear_history(ud)
            send_message(vk, user_id, "🧹 История очищена.", get_main_keyboard())
            return
        elif text == '/info':
            info_text = get_session_info(ud)
            send_message(vk, user_id, info_text, get_main_keyboard())
            return
        else:
            send_message(vk, user_id, "Неизвестная команда. Используйте кнопки или /help.", get_main_keyboard())
            return

    # Обработка кнопок
    if text in ['📄 Новый диалог', '📜 История', 'ℹ️ Информация', '🧹 Очистить историю', '❓ Помощь']:
        if handle_button(ud, text, user_id, vk):
            return

    # Состояние WAITING_DOC
    if ud["state"] == WAITING_DOC:
        if not event.attachments:
            send_message(vk, user_id, "Пожалуйста, отправьте файл в формате PDF или DOCX.", get_main_keyboard())
            return

        # Получение информации о документе
        try:
            msg_info = vk.messages.getById(message_ids=event.message_id)
            items = msg_info.get('items', [])
            if not items:
                send_message(vk, user_id, "Не удалось получить информацию о сообщении.", get_main_keyboard())
                return

            attachments = items[0].get('attachments', [])
            if not attachments:
                send_message(vk, user_id, "В сообщении нет вложений.", get_main_keyboard())
                return

            doc_info = None
            for att in attachments:
                if att.get('type') == 'doc':
                    doc_info = att.get('doc')
                    break

            if not doc_info:
                send_message(vk, user_id, "Вложенный файл не является документом.", get_main_keyboard())
                return

            ext = doc_info.get('ext')
            if ext not in ('pdf', 'docx'):
                send_message(vk, user_id, "Файл должен быть в формате PDF или DOCX.", get_main_keyboard())
                return

            doc_url = doc_info.get('url')
            doc_title = doc_info.get('title')

            if not doc_url:
                send_message(vk, user_id, "Не удалось получить ссылку на файл.", get_main_keyboard())
                return

        except Exception as e:
            logger.exception("Ошибка при получении информации о документе")
            send_message(vk, user_id, "Ошибка при обработке документа.", get_main_keyboard())
            return

        # Сообщение о начале индексации
        send_message(vk, user_id, "📄 Индексирую документ... Это займёт несколько секунд.", get_main_keyboard())

        # Скачивание файла
        send_typing(vk, user_id)
        try:
            resp = requests.get(doc_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                send_message(vk, user_id, "Не удалось скачать файл.", get_main_keyboard())
                return
            file_bytes = resp.content
        except Exception as e:
            logger.error(f"User {user_id}: download error: {e}")
            send_message(vk, user_id, "Ошибка при скачивании файла.", get_main_keyboard())
            return

        if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            send_message(vk, user_id, f"Файл слишком большой. Максимум {MAX_FILE_SIZE_MB} МБ.", get_main_keyboard())
            return

        session_id, status, error = backend_upload(file_bytes, doc_title, user_id)
        if session_id is None:
            if status == 408:
                send_message(vk, user_id, "Тайм-аут при загрузке файла. Попробуйте позже.", get_main_keyboard())
            elif status == 503:
                send_message(vk, user_id, "Сервер недоступен. Попробуйте позже.", get_main_keyboard())
            else:
                send_message(vk, user_id, f"Ошибка при загрузке файла: {error}", get_main_keyboard())
            return

        ud["session_id"] = session_id
        ud["filename"] = doc_title
        ud["session_created"] = time.time()
        clear_history(ud)
        ud["state"] = WAITING_QUESTION

        send_message(vk, user_id, f"✅ Документ «{doc_title}» загружен. Теперь задавайте вопросы.", get_main_keyboard())

        # Примеры вопросов
        examples = [
            "Каков максимальный размер гранта?",
            "Какие требования к заявителю?",
            "Каковы сроки подачи заявок?"
        ]
        examples_text = "💡 Примеры вопросов:\n" + "\n".join(f"• {q}" for q in examples)
        send_message(vk, user_id, examples_text, get_main_keyboard())
        return

    # Состояние WAITING_QUESTION
    if ud["state"] == WAITING_QUESTION:
        if not text:
            send_message(vk, user_id, "Вопрос не может быть пустым.", get_main_keyboard())
            return
        if len(text) > MAX_QUESTION_LENGTH:
            send_message(vk, user_id, f"Вопрос слишком длинный (макс. {MAX_QUESTION_LENGTH} символов).", get_main_keyboard())
            return
        if not check_rate_limit(ud):
            send_message(vk, user_id, "Слишком много вопросов. Подождите немного.", get_main_keyboard())
            return

        session_id = ud.get("session_id")
        if not session_id:
            send_message(vk, user_id, "Нет активной сессии. Начните сначала с кнопки 'Новый диалог'.", get_main_keyboard())
            ud["state"] = WAITING_DOC
            return

        send_typing(vk, user_id)
        add_to_history(ud, "user", text)

        answer, status, error = backend_ask(session_id, text, user_id)
        if status == 404:
            send_message(vk, user_id, "Сессия истекла. Загрузите документ заново кнопкой 'Новый диалог'.", get_main_keyboard())
            ud.pop("session_id", None)
            ud.pop("filename", None)
            ud.pop("session_created", None)
            ud["state"] = WAITING_DOC
            return
        if status != 200 or answer is None:
            if status == 408:
                send_message(vk, user_id, "Тайм-аут при запросе. Попробуйте ещё раз.", get_main_keyboard())
            elif status == 503:
                send_message(vk, user_id, "Сервер недоступен. Попробуйте позже.", get_main_keyboard())
            else:
                send_message(vk, user_id, f"Ошибка: {error}", get_main_keyboard())
            return

        if not answer.strip():
            answer = "[Бэкенд вернул пустой ответ]"
        add_to_history(ud, "assistant", answer)
        send_message(vk, user_id, answer, get_main_keyboard())
        return

    send_message(vk, user_id, "Непонятная ситуация. Используйте кнопки или /start.", get_main_keyboard())


def main():
    if not VK_TOKEN:
        logger.error("VK_TOKEN не задан в .env")
        return

    vk_session = vk_api.VkApi(token=VK_TOKEN)
    vk = vk_session.get_api()
    longpoll = VkLongPoll(vk_session, wait=90)

    logger.info("VK-бот запущен. Ожидание сообщений...")

    ok, msg = check_backend_health()
    if ok:
        logger.info("Бэкенд доступен.")
    else:
        logger.warning(f"Бэкенд недоступен: {msg}")

    while True:
        try:
            for event in longpoll.listen():
                if event.type == VkEventType.MESSAGE_NEW and event.to_me:
                    process_message(event, vk)
        except Exception as e:
            logger.exception("Ошибка в Long Poll, переподключение через 5 секунд")
            time.sleep(5)
            longpoll = VkLongPoll(vk_session, wait=90)


if __name__ == "__main__":
    main()
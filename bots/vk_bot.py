#!/usr/bin/env python3
"""
VK-бот для RAG-сервиса грантовых документов.
Единый стиль, минимум дублирования, дружелюбный UI.
"""

import os
import logging
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.utils import get_random_id
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from dotenv import load_dotenv

# Загрузка переменных окружения
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
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", 5))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 1))

# Состояния
WAITING_DOC, WAITING_QUESTION = range(2)

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

user_data: Dict[int, Dict[str, Any]] = {}


# ==================== КЛАВИАТУРА ====================
def get_main_keyboard() -> VkKeyboard:
    """Клавиатура главного меню с чередованием цветов."""
    keyboard = VkKeyboard(one_time=False)
    keyboard.add_button('Новый диалог', color=VkKeyboardColor.PRIMARY)
    keyboard.add_button('История', color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button('Информация', color=VkKeyboardColor.SECONDARY)
    keyboard.add_button('Очистить историю', color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button('Помощь', color=VkKeyboardColor.PRIMARY)
    return keyboard


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def get_user(user_id: int) -> Dict[str, Any]:
    """Получить (или создать) запись о пользователе."""
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


def reset_user(user_id: int) -> None:
    """Полный сброс данных пользователя."""
    user_data[user_id] = {
        "state": WAITING_DOC,
        "history": [],
        "session_id": None,
        "filename": None,
        "session_created": None,
        "last_question_time": 0
    }


def add_to_history(ud: Dict[str, Any], role: str, content: str) -> None:
    """Добавить запись в историю диалога."""
    history = ud.setdefault("history", [])
    history.append({"role": role, "content": content, "time": time.time()})
    if len(history) > MAX_HISTORY_STORED:
        ud["history"] = history[-MAX_HISTORY_STORED:]


def check_rate_limit(ud: Dict[str, Any]) -> bool:
    """Проверка частоты вопросов."""
    now = time.time()
    if now - ud.get("last_question_time", 0) < RATE_LIMIT_SECONDS:
        return False
    ud["last_question_time"] = now
    return True


def format_history(ud: Dict[str, Any]) -> str:
    """Форматирование истории для показа."""
    history = ud.get("history", [])
    if not history:
        return "История пуста."
    entries = []
    for entry in history[-MAX_HISTORY_SHOWN:]:
        dt = datetime.fromtimestamp(entry["time"]).strftime("%H:%M")
        prefix = "Вы" if entry["role"] == "user" else "Бот"
        entries.append(f"{prefix} [{dt}]: {entry['content']}")
    return "\n\n".join(entries)


def get_session_info(ud: Dict[str, Any]) -> str:
    """Информация о текущей сессии."""
    filename = ud.get("filename", "неизвестно")
    created = ud.get("session_created")
    if created is None:
        return f"Документ: {filename}\nВремя создания сессии неизвестно."
    elapsed = time.time() - created
    remaining = max(0, SESSION_TIMEOUT - elapsed)
    remaining_str = str(timedelta(seconds=int(remaining)))
    return f"Документ: {filename}\nОсталось времени сессии: {remaining_str}"


# ==================== ОТПРАВКА СООБЩЕНИЙ ====================

def send_message(vk: Any, user_id: int, text: str,
                 keyboard: Optional[VkKeyboard] = None) -> None:
    """Отправка сообщения с клавиатурой."""
    params = {
        "user_id": user_id,
        "message": text,
        "random_id": get_random_id()
    }
    if keyboard:
        params["keyboard"] = keyboard.get_keyboard()
    vk.messages.send(**params)


def send_typing(vk: Any, user_id: int) -> None:
    """Индикация 'печатает'."""
    try:
        vk.messages.setActivity(user_id=user_id, type="typing")
    except Exception:
        pass


# ==================== БЭКЕНД-ЗАПРОСЫ ====================

def backend_upload(file_bytes: bytes, filename: str, user_id: int
                   ) -> Tuple[Optional[str], Optional[int], str]:
    """Загрузка файла на бэкенд."""
    files = {'file': (filename, file_bytes)}
    try:
        resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
        if resp.status_code == 200:
            result = resp.json()
            session_id = result.get("session_id")
            if isinstance(session_id, str):
                return session_id, 200, ""
            else:
                return None, 200, "Invalid response format"
        else:
            logger.error(f"User {user_id}: Upload failed with status {resp.status_code}")
            return None, resp.status_code, f"HTTP {resp.status_code}"
    except requests.Timeout:
        return None, 408, "Request timeout"
    except requests.ConnectionError:
        return None, 503, "Connection refused"
    except Exception as e:
        logger.exception(f"User {user_id}: Upload error")
        return None, 500, str(e)


def backend_ask(session_id: str, question: str, user_id: int,
                retry: int = 0) -> Tuple[Optional[str], Optional[int], str]:
    """Запрос к бэкенду с вопросом. При ошибках 5xx повторяет запрос."""
    payload = {"session_id": session_id, "question": question}
    try:
        resp = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=120)
        if resp.status_code == 200:
            result = resp.json()
            answer = result.get("answer")
            if isinstance(answer, str):
                return answer, 200, ""
            else:
                return None, 200, "Invalid response format"
        elif resp.status_code >= 500 and retry < MAX_RETRIES:
            logger.warning(f"User {user_id}: Ask got {resp.status_code}, "
                           f"retrying {retry+1}/{MAX_RETRIES}")
            time.sleep(RETRY_DELAY * (retry + 1))
            return backend_ask(session_id, question, user_id, retry + 1)
        else:
            logger.error(f"User {user_id}: Ask failed with status {resp.status_code}")
            return None, resp.status_code, f"HTTP {resp.status_code}"
    except requests.Timeout:
        return None, 408, "Request timeout"
    except requests.ConnectionError:
        return None, 503, "Connection refused"
    except Exception as e:
        logger.exception(f"User {user_id}: Ask error")
        return None, 500, str(e)


# ==================== ОБРАБОТЧИКИ КОМАНД И КНОПОК ====================

def handle_button(vk: Any, user_id: int, ud: Dict[str, Any], text: str) -> bool:
    """
    Обработать нажатие кнопки меню.
    Возвращает True, если кнопка была обработана.
    """
    if text == 'Новый диалог':
        reset_user(user_id)
        send_message(vk, user_id,
                     "Начинаем новый диалог. Отправьте PDF или DOCX файл.",
                     get_main_keyboard())
        return True
    elif text == 'История':
        send_message(vk, user_id, format_history(ud), get_main_keyboard())
        return True
    elif text == 'Информация':
        send_message(vk, user_id, get_session_info(ud), get_main_keyboard())
        return True
    elif text == 'Очистить историю':
        ud["history"] = []
        send_message(vk, user_id, "История очищена.", get_main_keyboard())
        return True
    elif text == 'Помощь':
        show_help(vk, user_id)
        return True
    return False


def show_help(vk: Any, user_id: int) -> None:
    """Вывод справки."""
    text = (
        "Справка\n\n"
        "1. Отправьте PDF или DOCX файл (до 20 МБ).\n"
        "2. После загрузки задавайте вопросы по документу.\n"
        "3. Сессия длится 1 час после последнего вопроса.\n\n"
        "Команды:\n"
        "/start - начало работы\n"
        "/new - новый диалог\n"
        "/history - история сообщений\n"
        "/clear - очистить историю\n"
        "/info - информация о документе\n"
        "/help - эта справка\n"
        "/cancel - завершить диалог"
    )
    send_message(vk, user_id, text, get_main_keyboard())


# ==================== ГЛАВНЫЙ ДИСПЕТЧЕР ====================

def process_message(event: vk_api.longpoll.Event, vk: Any) -> None:
    """Главный диспетчер сообщений."""
    user_id = event.user_id
    raw_text = event.text.strip() if event.text else ""
    text = raw_text  # исходный текст для кнопок и вопросов

    logger.info(f"process_message: user={user_id}, "
                f"text={raw_text[:50] if raw_text else '<no text>'}")

    ud = get_user(user_id)

    # --- Обработка кнопок меню (точное совпадение) ---
    if text in ['Новый диалог', 'История', 'Информация',
                'Очистить историю', 'Помощь']:
        handle_button(vk, user_id, ud, text)
        return

    # --- Обработка команд ---
    if raw_text.startswith('/'):
        cmd = raw_text.split()[0] if raw_text else ''
        if cmd == '/start':
            reset_user(user_id)
            send_message(vk, user_id,
                         "Добро пожаловать! Я помогу вам разобраться в грантовой документации.\n"
                         "Отправьте PDF или DOCX файл, и я отвечу на ваши вопросы.",
                         get_main_keyboard())
        elif cmd == '/help':
            show_help(vk, user_id)
        elif cmd == '/new':
            reset_user(user_id)
            send_message(vk, user_id,
                         "Начинаем новый диалог. Отправьте PDF или DOCX файл.",
                         get_main_keyboard())
        elif cmd == '/history':
            send_message(vk, user_id, format_history(ud), get_main_keyboard())
        elif cmd == '/clear':
            ud["history"] = []
            send_message(vk, user_id, "История очищена.", get_main_keyboard())
        elif cmd == '/info':
            send_message(vk, user_id, get_session_info(ud), get_main_keyboard())
        elif cmd == '/cancel':
            ud["state"] = WAITING_DOC
            send_message(vk, user_id,
                         "Диалог завершён. Чтобы начать заново, отправьте /new "
                         "или нажмите кнопку «Новый диалог».",
                         get_main_keyboard())
        else:
            send_message(vk, user_id,
                         "Неизвестная команда. Используйте кнопки меню или /help.",
                         get_main_keyboard())
        return

    # --- Универсальная реакция на нетекстовые сообщения (без текста и без документа) ---
    if not raw_text and not event.attachments:
        send_message(vk, user_id,
                     "Я понимаю только текст и документы PDF/DOCX. "
                     "Используйте кнопки или команду /help.",
                     get_main_keyboard())
        return

    # --- Обработка вложений (только в WAITING_DOC) ---
    if event.attachments:
        # В состоянии WAITING_QUESTION вложения не принимаем
        if ud["state"] == WAITING_QUESTION:
            send_message(vk, user_id,
                         "Вы отправили документ, но у вас уже есть активная сессия.\n"
                         "Чтобы начать работу с новым документом, используйте /new "
                         "или кнопку «Новый диалог».",
                         get_main_keyboard())
            return

        # В состоянии WAITING_DOC обрабатываем строго документы PDF/DOCX
        if ud["state"] == WAITING_DOC:
            try:
                msg_info = vk.messages.getById(message_ids=event.message_id)
                items = msg_info.get('items', [])
                if not items:
                    send_message(vk, user_id,
                                 "Не удалось получить информацию о сообщении.",
                                 get_main_keyboard())
                    return
                attachments = items[0].get('attachments', [])
                doc_info = None
                for att in attachments:
                    if att.get('type') == 'doc':
                        doc_info = att.get('doc')
                        break
                if not doc_info:
                    send_message(vk, user_id,
                                 "Я понимаю только текст и документы PDF/DOCX. "
                                 "Используйте кнопки или команду /help.",
                                 get_main_keyboard())
                    return

                ext = doc_info.get('ext')
                if ext not in ('pdf', 'docx'):
                    send_message(vk, user_id,
                                 "Я понимаю только текст и документы PDF/DOCX. "
                                 "Используйте кнопки или команду /help.",
                                 get_main_keyboard())
                    return
                doc_url = doc_info.get('url')
                doc_title = doc_info.get('title')
                if not doc_url:
                    send_message(vk, user_id,
                                 "Не удалось получить ссылку на файл.",
                                 get_main_keyboard())
                    return

            except Exception as e:
                logger.exception(f"User {user_id}: Failed to get document info")
                send_message(vk, user_id,
                             "Ошибка при получении информации о файле.",
                             get_main_keyboard())
                return

            # Скачивание файла
            send_typing(vk, user_id)
            try:
                resp = requests.get(doc_url, timeout=30,
                                    headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code != 200:
                    send_message(vk, user_id, "Не удалось скачать файл.",
                                 get_main_keyboard())
                    return
                file_bytes = resp.content
            except Exception as e:
                logger.error(f"User {user_id}: download error: {e}")
                send_message(vk, user_id, "Ошибка при скачивании файла.",
                             get_main_keyboard())
                return

            if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
                send_message(vk, user_id,
                             f"Файл слишком большой. Максимум {MAX_FILE_SIZE_MB} МБ.",
                             get_main_keyboard())
                return

            # Загрузка на бэкенд
            send_message(vk, user_id,
                         "Индексирую документ... Это займёт несколько секунд.",
                         get_main_keyboard())
            session_id, status, error = backend_upload(file_bytes, doc_title, user_id)
            if session_id is None:
                if status in (408, 503):
                    send_message(vk, user_id,
                                 "Сервер временно недоступен. Попробуйте позже.",
                                 get_main_keyboard())
                else:
                    send_message(vk, user_id,
                                 "Ошибка обработки документа. Попробуйте ещё раз.",
                                 get_main_keyboard())
                return

            # Успешная загрузка
            ud["session_id"] = session_id
            ud["filename"] = doc_title
            ud["session_created"] = time.time()
            ud["history"] = []
            ud["state"] = WAITING_QUESTION

            send_message(vk, user_id,
                         f"Документ «{doc_title}» загружен. Теперь задавайте вопросы.",
                         get_main_keyboard())

            examples = (
                "Примеры вопросов:\n"
                "- Каков максимальный размер гранта?\n"
                "- Какие требования к заявителю?\n"
                "- Каковы сроки подачи заявок?"
            )
            send_message(vk, user_id, examples, get_main_keyboard())
            return

    # --- Обработка текстовых сообщений в зависимости от состояния ---
    if ud["state"] == WAITING_DOC:
        send_message(vk, user_id,
                     "Пожалуйста, отправьте файл PDF или DOCX. "
                     "Если хотите начать заново, используйте /new.",
                     get_main_keyboard())
        return

    if ud["state"] == WAITING_QUESTION:
        if not raw_text:
            send_message(vk, user_id, "Вопрос не может быть пустым.",
                         get_main_keyboard())
            return
        if len(raw_text) > MAX_QUESTION_LENGTH:
            send_message(vk, user_id,
                         f"Вопрос слишком длинный (макс. {MAX_QUESTION_LENGTH} символов).",
                         get_main_keyboard())
            return
        if not check_rate_limit(ud):
            send_message(vk, user_id, "Слишком много вопросов. Подождите немного.",
                         get_main_keyboard())
            return

        session_id = ud.get("session_id")
        if not session_id:
            send_message(vk, user_id,
                         "Нет активной сессии. Начните сначала: отправьте /new "
                         "или нажмите кнопку «Новый диалог».",
                         get_main_keyboard())
            ud["state"] = WAITING_DOC
            return

        send_typing(vk, user_id)
        add_to_history(ud, "user", raw_text)

        answer, status, error = backend_ask(session_id, raw_text, user_id)
        if status == 404:
            send_message(vk, user_id,
                         "Сессия истекла. Загрузите документ заново: /new "
                         "или кнопка «Новый диалог».",
                         get_main_keyboard())
            ud.pop("session_id", None)
            ud.pop("filename", None)
            ud.pop("session_created", None)
            ud["state"] = WAITING_DOC
            return
        if status != 200 or answer is None:
            if status in (408, 503):
                send_message(vk, user_id,
                             "Сервер временно недоступен. Попробуйте позже.",
                             get_main_keyboard())
            else:
                send_message(vk, user_id,
                             "Произошла ошибка. Попробуйте ещё раз.",
                             get_main_keyboard())
            return

        if not answer.strip():
            answer = "Информация не найдена"
        add_to_history(ud, "assistant", answer)
        send_message(vk, user_id, answer, get_main_keyboard())
        return

    # Неизвестное состояние – сброс
    reset_user(user_id)
    send_message(vk, user_id,
                 "Возникла непонятная ситуация. Начнём заново. Отправьте документ.",
                 get_main_keyboard())


# ==================== ЗАПУСК БОТА ====================

def main() -> None:
    """Главная точка входа."""
    if not VK_TOKEN:
        logger.error("VK_TOKEN не задан в .env")
        return

    vk_session = vk_api.VkApi(token=VK_TOKEN)
    vk = vk_session.get_api()
    longpoll = VkLongPoll(vk_session, wait=90)

    logger.info("VK-бот запущен. Ожидание сообщений...")

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
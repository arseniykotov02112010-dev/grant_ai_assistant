"""
СКРИПТ НЕ АДАПТИРОВАН ПОД ИЗМЕНЕННЫЕ main.py, config.py И ТД
ВСВЯЗИ С СЛОЖНОСТЬЮ РАБОТЫ В ТЕЛЕГРАММЕ
"""

import os
import logging
import time
import asyncio
import signal
import zipfile
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import html
import io  # для is_docx_content

from telegram import Update, Document
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ConversationHandler,
    ContextTypes,
    PicklePersistence,
)
import aiohttp
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла в папке bots
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# ==================== КОНФИГУРАЦИЯ ИЗ ОКРУЖЕНИЯ ====================
BOT_TOKEN = os.getenv("BOT_TOKEN")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", 3600))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 20))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", 1000))
MAX_HISTORY_STORED = int(os.getenv("MAX_HISTORY_STORED", 200))
MAX_HISTORY_SHOWN = int(os.getenv("MAX_HISTORY_SHOWN", 20))
PERSISTENCE_FILE = os.getenv("PERSISTENCE_FILE", "bot_data.pickle")
MAX_FILENAME_LENGTH = int(os.getenv("MAX_FILENAME_LENGTH", 255))
MAX_MESSAGE_LENGTH = 4096  # лимит Telegram
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", 2))  # мин. интервал между вопросами
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))  # повторные попытки при 5xx
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 1))  # секунд между повторами

# Состояния для ConversationHandler
WAITING_DOC, WAITING_QUESTION = range(2)

# Разрешённые расширения и MIME-типы
ALLOWED_EXTENSIONS = ('.pdf', '.docx')
ALLOWED_MIME_TYPES = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Глобальная HTTP-сессия
http_session: Optional[aiohttp.ClientSession] = None


# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С БЭКЕНДОМ ====================

async def check_backend_health() -> Tuple[bool, str]:
    """Проверяет доступность бэкенда через /health."""
    try:
        async with http_session.get(f"{BACKEND_URL}/health", timeout=5) as resp:
            if resp.status == 200:
                return True, "OK"
            else:
                return False, f"HTTP {resp.status}"
    except asyncio.TimeoutError:
        return False, "timeout"
    except aiohttp.ClientConnectorError:
        return False, "connection refused"
    except Exception as e:
        return False, str(e)


async def backend_upload(file_bytes: bytes, filename: str, user_id: int) -> Tuple[Optional[str], Optional[int], str]:
    """
    Отправляет файл на /upload бэкенда.
    Возвращает (session_id, status_code, error_message).
    """
    data = aiohttp.FormData()
    data.add_field('file', file_bytes, filename=filename)
    try:
        async with http_session.post(f"{BACKEND_URL}/upload", data=data, timeout=120) as resp:
            if resp.status == 200:
                try:
                    result = await resp.json()
                except Exception as e:
                    logger.error(f"User {user_id}: Failed to parse JSON from upload response: {e}")
                    return None, 500, "Invalid JSON response from server"
                session_id = result.get("session_id")
                if isinstance(session_id, str):
                    return session_id, 200, ""
                else:
                    return None, 200, "Invalid response format (missing session_id)"
            else:
                text = await resp.text()
                logger.error(f"User {user_id}: Upload error {resp.status}: {text[:200]}")
                return None, resp.status, f"HTTP {resp.status}"
    except asyncio.TimeoutError:
        logger.error(f"User {user_id}: Upload timeout")
        return None, 408, "Request timeout"
    except aiohttp.ClientConnectorError as e:
        logger.error(f"User {user_id}: Cannot connect to backend: {e}")
        return None, 503, "Cannot connect to backend"
    except Exception as e:
        logger.exception(f"User {user_id}: Unexpected error in backend_upload")
        return None, 500, str(e)


async def backend_ask(session_id: str, question: str, user_id: int, retry: int = 0) -> Tuple[Optional[str], Optional[int], str]:
    """
    Отправляет вопрос на /ask бэкенда с поддержкой повторных попыток при 5xx.
    Возвращает (answer, status_code, error_message).
    """
    payload = {"session_id": session_id, "question": question}
    try:
        async with http_session.post(f"{BACKEND_URL}/ask", json=payload, timeout=300) as resp:
            if resp.status == 200:
                try:
                    result = await resp.json()
                except Exception as e:
                    logger.error(f"User {user_id}: Failed to parse JSON from ask response: {e}")
                    return None, 500, "Invalid JSON response from server"
                answer = result.get("answer")
                if isinstance(answer, str):
                    return answer, 200, ""
                else:
                    return None, 200, "Invalid response format (missing or non-string answer)"
            elif resp.status >= 500 and retry < MAX_RETRIES:
                # Повтор при ошибках сервера
                logger.warning(f"User {user_id}: Server error {resp.status}, retry {retry+1}/{MAX_RETRIES}")
                await asyncio.sleep(RETRY_DELAY * (retry + 1))
                return await backend_ask(session_id, question, user_id, retry + 1)
            else:
                text = await resp.text()
                logger.error(f"User {user_id}: Ask error {resp.status}: {text[:200]}")
                return None, resp.status, f"HTTP {resp.status}"
    except asyncio.TimeoutError:
        logger.error(f"User {user_id}: Ask timeout")
        return None, 408, "Request timeout"
    except aiohttp.ClientConnectorError as e:
        logger.error(f"User {user_id}: Cannot connect to backend: {e}")
        return None, 503, "Cannot connect to backend"
    except Exception as e:
        logger.exception(f"User {user_id}: Unexpected error in backend_ask")
        return None, 500, str(e)


# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С ИСТОРИЕЙ ====================

def init_history() -> list:
    return []

def add_to_history(user_data: Dict[str, Any], role: str, content: str):
    if "history" not in user_data:
        user_data["history"] = init_history()
    user_data["history"].append({
        "role": role,
        "content": content,
        "time": time.time()
    })
    # обрезаем до MAX_HISTORY_STORED
    if len(user_data["history"]) > MAX_HISTORY_STORED:
        user_data["history"] = user_data["history"][-MAX_HISTORY_STORED:]

def clear_history(user_data: Dict[str, Any]):
    user_data["history"] = init_history()


def format_history_entry(entry: Dict[str, Any]) -> str:
    """Форматирует одну запись для отображения."""
    dt = datetime.fromtimestamp(entry["time"]).strftime("%H:%M")
    prefix = "👤" if entry["role"] == "user" else "🤖"
    return f"{prefix} [{dt}] {entry['content']}"


def get_history_text(user_data: Dict[str, Any]) -> str:
    """Возвращает текст последних MAX_HISTORY_SHOWN записей."""
    history = user_data.get("history")
    if not history:
        return "📭 История пуста."
    entries = [format_history_entry(e) for e in list(history)[-MAX_HISTORY_SHOWN:]]
    return "\n\n".join(entries)


def clear_history(user_data: Dict[str, Any]):
    """Очищает историю."""
    user_data["history"] = init_history()


def get_session_info(user_data: Dict[str, Any]) -> str:
    """Возвращает информацию о текущей сессии."""
    filename = user_data.get("filename", "неизвестно")
    created = user_data.get("session_created")
    if created is None:
        return f"📄 Документ: {filename}\n⏱️ Время создания сессии неизвестно."
    elapsed = time.time() - created
    remaining = max(0, SESSION_TIMEOUT - elapsed)
    remaining_str = str(timedelta(seconds=int(remaining)))
    return f"📄 Документ: {filename}\n⏱️ Осталось времени сессии: {remaining_str}"


def check_rate_limit(user_data: Dict[str, Any]) -> bool:
    """Проверяет, не слишком ли часто задаются вопросы."""
    last_question_time = user_data.get("last_question_time", 0)
    if time.time() - last_question_time < RATE_LIMIT_SECONDS:
        return False
    user_data["last_question_time"] = time.time()
    return True


# ==================== ПРОВЕРКИ ФАЙЛОВ ====================

def is_pdf_content(file_bytes: bytes) -> bool:
    """Проверяет сигнатуру PDF (допускается смещение)."""
    return file_bytes.lstrip(b'\x00').startswith(b'%PDF')


def is_docx_content(file_bytes: bytes) -> bool:
    """Проверяет, является ли содержимое валидным DOCX (ZIP-архив с [Content_Types].xml)."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            return '[Content_Types].xml' in z.namelist()
    except Exception:
        return False


# ==================== ОБРАБОТЧИКИ КОМАНД И СООБЩЕНИЙ ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Команда /start – начало диалога, полный сброс."""
    user = update.effective_user
    context.user_data.clear()
    await update.message.reply_text(
        f"Привет, {html.escape(user.first_name)}!\n"
        "Я помогу тебе найти информацию в грантовых документах.\n"
        "Отправь мне PDF или DOCX файл с конкурсной документацией, и я буду отвечать на вопросы.\n\n"
        "<b>Команды:</b>\n"
        "/new – начать заново (сбросить текущий диалог)\n"
        "/history – показать историю сообщений\n"
        "/clear – очистить историю (но оставить текущий документ)\n"
        "/info – информация о текущем документе\n"
        "/help – показать это сообщение\n"
        "/cancel – выйти из диалога",
        parse_mode=ParseMode.HTML
    )
    return WAITING_DOC


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /help."""
    await update.message.reply_text(
        f"<b>Справка</b>\n\n"
        f"1. Отправь мне PDF или DOCX файл (до {MAX_FILE_SIZE_MB} МБ).\n"
        f"2. После загрузки задавай вопросы по документу.\n"
        f"3. Сессия живёт {SESSION_TIMEOUT // 60} минут после последнего вопроса. Если время вышло, начни заново с /new.\n\n"
        "<b>Команды:</b>\n"
        "/start – начало работы\n"
        "/new – сбросить текущий диалог и загрузить новый документ\n"
        "/history – показать последние {} сообщений\n"
        "/clear – очистить историю\n"
        "/info – информация о текущем документе\n"
        "/help – эта справка\n"
        "/cancel – выйти из диалога".format(MAX_HISTORY_SHOWN),
        parse_mode=ParseMode.HTML
    )


async def new_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Команда /new – сброс диалога, предложение загрузить новый документ."""
    context.user_data.clear()
    await update.message.reply_text(
        "🔄 Начинаем новый диалог. Отправь мне PDF или DOCX файл."
    )
    return WAITING_DOC


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /history – показать историю."""
    text = get_history_text(context.user_data)
    # Разбивка на части, если длинное
    for i in range(0, len(text), MAX_MESSAGE_LENGTH):
        await update.message.reply_text(text[i:i+MAX_MESSAGE_LENGTH])


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /clear – очистить историю."""
    clear_history(context.user_data)
    await update.message.reply_text("🧹 История очищена.")


async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /info – информация о текущем документе."""
    text = get_session_info(context.user_data)
    await update.message.reply_text(text)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Команда /cancel – выход из диалога."""
    await update.message.reply_text(
        "Диалог завершён. Чтобы начать заново, отправь /start."
    )
    return ConversationHandler.END


async def prompt_for_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает текст в состоянии WAITING_DOC – напоминает, что нужен документ."""
    await update.message.reply_text(
        "📄 Пожалуйста, отправьте PDF или DOCX файл. Если хотите начать заново, используйте /new."
    )
    return WAITING_DOC


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обрабатывает загруженный документ (PDF или DOCX):
    - проверяет расширение, MIME-тип, размер, содержимое
    - отправляет на бэкенд
    - сохраняет session_id, имя файла, время создания
    - очищает историю
    - переходит в состояние ожидания вопроса
    """
    user_id = update.effective_user.id
    document: Document = update.message.document
    filename = document.file_name

    # Проверка длины имени файла
    if len(filename) > MAX_FILENAME_LENGTH:
        await update.message.reply_text(f"❌ Имя файла слишком длинное (максимум {MAX_FILENAME_LENGTH} символов).")
        return WAITING_DOC

    # Проверка расширения
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        await update.message.reply_text("❌ Пожалуйста, отправь файл в формате PDF или DOCX.")
        return WAITING_DOC

    # Проверка MIME-типа, если он присутствует
    if document.mime_type and document.mime_type not in ALLOWED_MIME_TYPES.values():
        await update.message.reply_text("❌ Тип файла не соответствует PDF или DOCX.")
        return WAITING_DOC

    # Проверка размера
    if document.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await update.message.reply_text(f"❌ Файл слишком большой. Максимальный размер: {MAX_FILE_SIZE_MB} МБ.")
        return WAITING_DOC

    await update.message.chat.send_action(action="typing")

    # Скачиваем файл с таймаутом
    try:
        file = await context.bot.get_file(document.file_id)
        file_bytes = await file.download_as_bytearray()
    except Exception as e:
        logger.error(f"User {user_id}: Error downloading file: {e}")
        await update.message.reply_text("❌ Ошибка при скачивании файла. Попробуй ещё раз.")
        return WAITING_DOC

    # Проверка на пустой файл
    if len(file_bytes) == 0:
        await update.message.reply_text("❌ Файл пуст.")
        return WAITING_DOC

    # Проверка сигнатуры в зависимости от типа
    if filename.lower().endswith('.pdf'):
        if not is_pdf_content(file_bytes):
            await update.message.reply_text("❌ Файл не является корректным PDF (отсутствует сигнатура %PDF).")
            return WAITING_DOC
    else:  # .docx
        if not is_docx_content(file_bytes):
            await update.message.reply_text("❌ Файл не является корректным DOCX (не является ZIP-архивом или отсутствует [Content_Types].xml).")
            return WAITING_DOC

    # Отправляем на бэкенд
    session_id, status, error = await backend_upload(file_bytes, filename, user_id)
    if session_id is None:
        if status == 408:
            await update.message.reply_text("⏱️ Тайм-аут при загрузке файла на сервер. Попробуй позже.")
        elif status == 503:
            await update.message.reply_text("🔌 Сервер недоступен. Убедись, что бэкенд запущен.")
        else:
            await update.message.reply_text(f"❌ Ошибка при загрузке файла (код {status}).")
        return WAITING_DOC

    # Сохраняем данные сессии
    context.user_data["session_id"] = session_id
    context.user_data["filename"] = filename
    context.user_data["session_created"] = time.time()
    context.user_data["history"] = init_history()  # новая история

    await update.message.reply_text(
        f"✅ Документ «{html.escape(filename)}» загружен и проиндексирован.\n"
        f"Теперь задавай вопросы. Для просмотра команд используй /help.",
        parse_mode=ParseMode.HTML
    )
    return WAITING_QUESTION


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Получает вопрос, проверяет длину, отправляет на бэкенд,
    обрабатывает ответ и обновляет историю.
    При ошибке 404 (сессия истекла) предлагает начать заново.
    """
    user_id = update.effective_user.id
    question = update.message.text.strip()

    # Проверка на пустой вопрос
    if not question:
        await update.message.reply_text("❌ Вопрос не может быть пустым.")
        return WAITING_QUESTION

    # Проверка длины вопроса
    if len(question) > MAX_QUESTION_LENGTH:
        await update.message.reply_text(
            f"❌ Вопрос слишком длинный (максимум {MAX_QUESTION_LENGTH} символов). Пожалуйста, сократи."
        )
        return WAITING_QUESTION

    # Проверка rate limit
    if not check_rate_limit(context.user_data):
        await update.message.reply_text("⏱️ Слишком много вопросов. Подожди немного.")
        return WAITING_QUESTION

    session_id = context.user_data.get("session_id")
    if not session_id:
        await update.message.reply_text(
            "❌ Нет активной сессии. Начни сначала с /start или /new."
        )
        return ConversationHandler.END

    await update.message.chat.send_action(action="typing")

    # Добавляем вопрос в историю
    add_to_history(context.user_data, "user", question)

    answer, status, error = await backend_ask(session_id, question, user_id)

    if status == 404:
        # Сессия истекла или не найдена
        await update.message.reply_text(
            "❌ Время сессии истекло. Пожалуйста, загрузи документ заново командой /new."
        )
        # Очищаем данные сессии, но историю оставляем
        context.user_data.pop("session_id", None)
        context.user_data.pop("filename", None)
        context.user_data.pop("session_created", None)
        return WAITING_DOC

    if status != 200 or answer is None:
        if status == 408:
            await update.message.reply_text("⏱️ Тайм-аут при запросе к серверу. Попробуй ещё раз.")
        elif status == 503:
            await update.message.reply_text("🔌 Сервер недоступен. Убедись, что бэкенд запущен.")
        else:
            await update.message.reply_text(f"❌ Ошибка при получении ответа (код {status}).")
        return WAITING_QUESTION

    # Если ответ пустой, заменяем на предупреждение
    if not answer.strip():
        answer = "[Бэкенд вернул пустой ответ]"
        add_to_history(context.user_data, "assistant", answer)  # добавляем, чтобы пользователь видел
    else:
        add_to_history(context.user_data, "assistant", answer)

    # Экранируем HTML-спецсимволы, чтобы не сломать разметку, и разбиваем на части
    safe_answer = html.escape(answer)
    for i in range(0, len(safe_answer), MAX_MESSAGE_LENGTH):
        part = safe_answer[i:i + MAX_MESSAGE_LENGTH]
        await update.message.reply_text(part)

    return WAITING_QUESTION


async def handle_document_in_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Если пользователь отправляет документ в состоянии WAITING_QUESTION,
    предлагаем начать новый диалог через /new.
    """
    await update.message.reply_text(
        "📄 Ты отправил новый документ, но у тебя уже есть активная сессия.\n"
        "Чтобы начать работу с новым документом, используй команду /new."
    )
    return WAITING_QUESTION


async def unsupported_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает любые медиа (кроме документов) внутри диалога."""
    await update.message.reply_text(
        "📄 Я принимаю только PDF или DOCX файлы. Если хочешь начать заново, используй /new."
    )


async def fallback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик сообщений вне диалога."""
    await update.message.reply_text(
        "Я не понимаю эту команду. Отправь /start, чтобы начать работу."
    )


# ==================== ЗАПУСК БОТА ====================

async def shutdown(signal, loop):
    """Закрываем HTTP-сессию при получении сигнала."""
    global http_session
    if http_session and not http_session.closed:
        await http_session.close()
        logger.info("HTTP session closed.")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    logger.info("Cancelling pending tasks...")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


async def post_shutdown(application: Application) -> None:
    """Закрываем HTTP-сессию при остановке приложения."""
    global http_session
    if http_session and not http_session.closed:
        await http_session.close()
        logger.info("HTTP session closed.")


async def startup_check(application: Application) -> None:
    """Проверяет доступность бэкенда при старте и создаёт HTTP-сессию."""
    global http_session
    # Устанавливаем общие таймауты для сессии
    timeout = aiohttp.ClientTimeout(total=300, connect=5, sock_read=300)
    http_session = aiohttp.ClientSession(timeout=timeout)

    ok, msg = await check_backend_health()
    if ok:
        logger.info("Backend health check passed.")
    else:
        logger.warning(f"Backend health check failed: {msg}. Bot will continue, but file upload may fail.")


def main() -> None:
    """Запуск бота."""
    if not BOT_TOKEN:
        print("❌ ОШИБКА: Укажите токен бота в переменной BOT_TOKEN в файле .env")
        return

    # Создаём persistence с версионированием
    persistence = PicklePersistence(filepath=PERSISTENCE_FILE, single_file=True, context_types={})

    # Создаём приложение
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .persistence(persistence)
        .post_init(startup_check)
        .post_shutdown(post_shutdown)
        .build()
    )

    # Обработчик диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            WAITING_DOC: [
                MessageHandler(filters.Document.ALL, handle_document),
                MessageHandler(filters.TEXT & ~filters.COMMAND, prompt_for_document),
                MessageHandler(filters.PHOTO | filters.VIDEO | filters.AUDIO | filters.VOICE | filters.Sticker.ALL, unsupported_media),
                CommandHandler("new", new_dialog),
                CommandHandler("help", help_command),
                CommandHandler("cancel", cancel),
            ],
            WAITING_QUESTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question),
                MessageHandler(filters.Document.ALL, handle_document_in_question),
                MessageHandler(filters.PHOTO | filters.VIDEO | filters.AUDIO | filters.VOICE | filters.Sticker.ALL, unsupported_media),
                CommandHandler("new", new_dialog),
                CommandHandler("history", history_command),
                CommandHandler("clear", clear_command),
                CommandHandler("info", info_command),
                CommandHandler("help", help_command),
                CommandHandler("cancel", cancel),
            ],
        },
        fallbacks=[
            CommandHandler("cancel", cancel),
            CommandHandler("start", start),
        ],
        name="grant_assistant_dialog",
        persistent=True,
    )

    application.add_handler(conv_handler)

    # Обработчики команд вне диалога
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("new", new_dialog))

    # Всё остальное
    application.add_handler(MessageHandler(filters.ALL, fallback))

    # Установка обработчиков сигналов для graceful shutdown
    loop = asyncio.get_event_loop()
    signals = (signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s, loop)))

    logger.info("Бот запущен. Нажми Ctrl+C для остановки.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
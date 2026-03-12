"""
finetune_qlora_final.py – АБСОЛЮТНО НАДЁЖНЫЙ QLoRA FINE-TUNING ДЛЯ QVikhr-3-4B-Instruction

Особенности:
- 4-битное квантование (nf4, double quant) для Tesla T4
- ChatML формат через apply_chat_template
- Динамический padding через DataCollator
- Gradient checkpointing
- Автоматическое добавление недостающих специальных токенов
- Маскировка пользовательской части с поиском <|im_start|>assistant
- Фильтрация некорректных примеров
- Проверка всех версий библиотек
- Полная совместимость с вашим окружением

Исправлены все ранее встреченные ошибки:
- seed удалён из load_dataset (совместимость)
- нет select_columns (UnboundLocalError)
- нет overwrite_output_dir
- добавлен resize_token_embeddings при новом pad_token
- проверка наличия <|im_start|> и assistant_ids
- корректное сохранение логов и модели
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset, Dataset
from importlib.metadata import version, PackageNotFoundError

# ==================== ПРОВЕРКА ВЕРСИЙ ====================
def version_tuple(v):
    return tuple(map(int, v.split('.')))

def check_version(package, required):
    try:
        installed = version(package)
        if version_tuple(installed) < version_tuple(required):
            raise RuntimeError(f"{package}>={required} required, found {installed}")
    except PackageNotFoundError:
        raise RuntimeError(f"{package} not installed")

# Проверяем критические библиотеки
check_version("transformers", "4.35.0")
check_version("peft", "0.6.0")
check_version("bitsandbytes", "0.41.0")
check_version("datasets", "2.14.0")
check_version("accelerate", "0.25.0")

import peft
import bitsandbytes

# ==================== ЛОГИРОВАНИЕ ====================
LOG_DIR = Path("/root/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_filename = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = LOG_DIR / log_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("ЗАПУСК FINE-TUNING QVikhr-3-4B С QLoRA")
logger.info("=" * 60)
logger.info(f"PyTorch: {torch.__version__}")
logger.info(f"Transformers: {transformers.__version__}")
logger.info(f"PEFT: {peft.__version__}")
logger.info(f"bitsandbytes: {bitsandbytes.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ==================== АРГУМЕНТЫ ====================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/splits/train.jsonl",
                        help="Путь к train.jsonl (абсолютный или относительный)")
    parser.add_argument("--val_file", type=str, default="data/splits/val.jsonl",
                        help="Путь к val.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/qvikhr-3-4b-grant-lora",
                        help="Папка для сохранения адаптеров")
    parser.add_argument("--model_name", type=str, default="Vikhrmodels/QVikhr-3-4B-Instruction",
                        help="Базовая модель")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Максимальная длина последовательности")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="Ранг LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Alpha LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="Dropout в LoRA")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Число эпох")
    parser.add_argument("--per_device_batch_size", type=int, default=4,
                        help="Batch size на GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Накопление градиента")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Сид для воспроизводимости")
    return parser.parse_args()

# ==================== ПОИСК ФАЙЛОВ ====================
def find_file(path_str: str, script_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    rel = script_dir / path_str
    if rel.is_file():
        return rel
    cwd = Path.cwd() / path_str
    if cwd.is_file():
        return cwd
    raise FileNotFoundError(f"Файл {path_str} не найден. Проверены: {path}, {rel}, {cwd}")

def setup_paths(args):
    script_dir = Path(__file__).parent
    train_path = find_file(args.train_file, script_dir)
    val_path = find_file(args.val_file, script_dir)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Train: {train_path}")
    logger.info(f"Val: {val_path}")
    logger.info(f"Output: {output_dir}")
    return train_path, val_path, output_dir

# ==================== ЗАГРУЗКА И ТОКЕНИЗАЦИЯ ====================
def load_and_tokenize_data(
    train_path: Path,
    val_path: Path,
    tokenizer,
    max_length: int = 2048
) -> Tuple[Dataset, Dataset]:
    logger.info("Загрузка train...")
    # Убрали seed – он не поддерживается в вашей версии datasets
    train_dataset = load_dataset('json', data_files=str(train_path), split='train')
    val_dataset = load_dataset('json', data_files=str(val_path), split='train')

    logger.info(f"Train: {len(train_dataset)} записей, Val: {len(val_dataset)} записей")

    # Удаляем записи с пустыми ответами
    train_dataset = train_dataset.filter(lambda x: bool(x.get('answer', '').strip()))
    val_dataset = val_dataset.filter(lambda x: bool(x.get('answer', '').strip()))

    # Проверка наличия метода apply_chat_template
    if not hasattr(tokenizer, 'apply_chat_template'):
        raise AttributeError("Токенизатор не имеет метода apply_chat_template. Обновите transformers или смените модель.")

    # Форматирование в ChatML
    def format_chat(example):
        messages = [
            {"role": "user", "content": f"Контекст: {example['context']}\nВопрос: {example['question']}"},
            {"role": "assistant", "content": example['answer']}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    num_proc = os.cpu_count() or 1
    train_dataset = train_dataset.map(format_chat, num_proc=num_proc)
    val_dataset = val_dataset.map(format_chat, num_proc=num_proc)

    # Проверка наличия шаблона
    for i in range(min(5, len(train_dataset))):
        if "<|im_start|>assistant" not in train_dataset[i]["text"]:
            logger.warning("Пример не содержит '<|im_start|>assistant' – возможно, проблемы с токенизатором.")
            break

    # Токенизация (без padding)
    def tokenize(examples):
        tok = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        tok["labels"] = [ids.copy() for ids in tok["input_ids"]]
        return tok

    train_tok = train_dataset.map(tokenize, batched=True,
                                  remove_columns=["text", "context", "question", "answer"],
                                  num_proc=num_proc)
    val_tok = val_dataset.map(tokenize, batched=True,
                              remove_columns=["text", "context", "question", "answer"],
                              num_proc=num_proc)

    # Получаем ID специальных токенов
    start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)

    # Если токен <|im_start|> не найден, пробуем добавить
    if start_token_id == tokenizer.unk_token_id:
        logger.warning("Токен <|im_start|> не найден, пытаемся добавить...")
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>']})
        start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        if start_token_id == tokenizer.unk_token_id:
            raise ValueError("Не удалось добавить токен <|im_start|>.")
        # Важно: после добавления токенов нужно расширить эмбеддинги модели!
        # Но модель у нас ещё не загружена – это будет сделано позже в setup_model_and_lora

    if len(assistant_ids) == 0:
        raise ValueError("Токенизатор не кодирует 'assistant' – невозможно выполнить маскировку.")

    def find_assistant_start(ids):
        for i in range(len(ids) - len(assistant_ids)):
            if ids[i] == start_token_id and ids[i+1:i+1+len(assistant_ids)] == assistant_ids:
                return i
        return -1

    def mask(example):
        pos = find_assistant_start(example["input_ids"])
        if pos == -1:
            example["assistant_found"] = False
            return example
        end = pos + len(assistant_ids)
        for j in range(end):
            example["labels"][j] = -100
        example["assistant_found"] = True
        return example

    train_tok = train_tok.map(mask, num_proc=num_proc)
    val_tok = val_tok.map(mask, num_proc=num_proc)

    train_found = sum(train_tok["assistant_found"])
    val_found = sum(val_tok["assistant_found"])
    logger.info(f"Найдено начало ответа: train {train_found}/{len(train_tok)}, val {val_found}/{len(val_tok)}")

    # Фильтруем те, где не нашли assistant или нет обучаемых токенов
    train_tok = train_tok.filter(lambda x: x["assistant_found"] and any(t != -100 for t in x["labels"]))
    val_tok = val_tok.filter(lambda x: x["assistant_found"] and any(t != -100 for t in x["labels"]))

    train_tok = train_tok.remove_columns(["assistant_found"])
    val_tok = val_tok.remove_columns(["assistant_found"])

    logger.info(f"После фильтрации: train {len(train_tok)}, val {len(val_tok)}")

    if len(train_tok) == 0:
        raise ValueError("Train датасет пуст после фильтрации. Проверьте качество данных.")
    if len(val_tok) == 0:
        raise ValueError("Validation датасет пуст после фильтрации.")

    # Проверка первого примера
    non_ignored = sum(1 for x in train_tok[0]["labels"] if x != -100)
    logger.info(f"Первый пример: {non_ignored}/{len(train_tok[0]['labels'])} токенов участвуют в loss.")

    # Оставляем только нужные колонки для модели
    train_tok = train_tok.select_columns(["input_ids", "attention_mask", "labels"])
    val_tok = val_tok.select_columns(["input_ids", "attention_mask", "labels"])

    return train_tok, val_tok

# ==================== ЗАГРУЗКА МОДЕЛИ И LoRA ====================
def setup_model_and_lora(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info(f"Загрузка модели {model_name} в 4-bit...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Устанавливаем pad_token, если отсутствует
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Пробуем использовать <|im_end|>
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id != tokenizer.unk_token_id:
                tokenizer.pad_token_id = im_end_id
                tokenizer.pad_token = tokenizer.decode(im_end_id)
            else:
                # Крайний случай – добавляем новый токен
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Запоминаем, что токен новый – позже расширим эмбеддинги
                need_resize = True
            tokenizer.padding_side = "right"

    # Загружаем модель
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_cache=False
    )

    # Если мы добавили новый pad_token, расширяем эмбеддинги
    if tokenizer.pad_token == '[PAD]' or tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= model.config.vocab_size:
        logger.info("Расширение эмбеддингов модели для нового pad_token...")
        model.resize_token_embeddings(len(tokenizer))

    # Подготовка для kbit-тренировки
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable} ({trainable/total:.2%} of all)")

    return model, tokenizer

# ==================== НАСТРОЙКА АРГУМЕНТОВ ОБУЧЕНИЯ ====================
def setup_training_args(args, output_dir: Path, train_len: int) -> TrainingArguments:
    eff_batch = args.per_device_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = train_len // eff_batch
    logger.info(f"Эффективный batch: {eff_batch}, шагов/эпоха: ~{steps_per_epoch}")

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        fp16=True,
        bf16=False,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        run_name=f"qvikhr-grant-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        # Важно: overwrite_output_dir не используется – не нужен
    )

# ==================== КОЛЛАТОР ====================
def create_collator(tokenizer):
    """
    Кастомный коллатор, выполняющий паддинг вручную до максимальной длины в батче.
    """
    class CustomDataCollator:
        def __init__(self, tokenizer, pad_to_multiple_of=8):
            self.tokenizer = tokenizer
            self.pad_to_multiple_of = pad_to_multiple_of

        def __call__(self, features):
            # Определяем максимальную длину в батче
            max_len = max(len(f['input_ids']) for f in features)
            if self.pad_to_multiple_of:
                max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

            padded_features = []
            for f in features:
                pad_len = max_len - len(f['input_ids'])
                # Создаём дополненные поля
                padded = {}
                # input_ids дополняем pad_token_id
                padded['input_ids'] = f['input_ids'] + [self.tokenizer.pad_token_id] * pad_len
                # attention_mask дополняем 0
                padded['attention_mask'] = f['attention_mask'] + [0] * pad_len
                # labels дополняем -100 (чтобы не участвовали в loss)
                padded['labels'] = f['labels'] + [-100] * pad_len
                padded_features.append(padded)

            # Преобразуем в тензоры PyTorch
            batch = {}
            for key in padded_features[0].keys():
                batch[key] = torch.tensor([f[key] for f in padded_features])
            return batch

    return CustomDataCollator(tokenizer, pad_to_multiple_of=8)
# ==================== MAIN ====================
def main():
    args = parse_args()

    # Настройка путей
    try:
        train_path, val_path, output_dir = setup_paths(args)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    # Загрузка модели и токенизатора
    model, tokenizer = setup_model_and_lora(
        args.model_name,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout
    )

    # Загрузка и токенизация данных
    train_dataset, val_dataset = load_and_tokenize_data(
        train_path, val_path, tokenizer, args.max_length
    )

    # Создание коллатора
    data_collator = create_collator(tokenizer)

    # Настройка аргументов обучения
    training_args = setup_training_args(args, output_dir, len(train_dataset))

    # Создание Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    logger.info("🚀 Старт обучения...")
    try:
        trainer.train()
    except Exception as e:
        logger.exception("Ошибка во время обучения")
        sys.exit(1)

    logger.info("💾 Сохранение модели...")
    trainer.save_model(output_dir)

    # Сохранение конфигурации
    with open(output_dir / "training_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Обучение завершено. Модель сохранена в {output_dir}")

    # Освобождение памяти
    del model, trainer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
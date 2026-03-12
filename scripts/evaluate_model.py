"""
evaluate_model.py – Финальная версия для оценки модели в трёх режимах:
1. Базовая модель без контекста.
2. Базовая модель с контекстом (RAG).
3. Дообученная модель с контекстом (RAG).

Инструкция по запуску скрипта evaluate_model.py:

1. Подключитесь к серверу с GPU:
   ssh root@<IP_сервера>

2. Активируйте виртуальное окружение (если используется):
   source /path/to/venv/bin/activate

3. Убедитесь, что тестовый файл (test.jsonl) содержит поля:
   doc_id, question, answer, context

4. Запустите оценку в трёх режимах:

   # Режим 1: базовая модель без контекста
   python evaluate_model.py --mode base --log_interval 15 --output_dir results_base

   # Режим 2: базовая модель + RAG (с контекстом)
   python evaluate_model.py --mode base_rag --log_interval 15 --output_dir results_base_rag

   # Режим 3: дообученная модель + RAG
   python evaluate_model.py --mode finetuned_rag --adapter_path /path/to/adapter \
       --log_interval 15 --output_dir results_finetuned_rag

   Для отладки добавьте флаг --debug (выводит контекст и примеры ответов).
   По умолчанию промежуточные результаты выводятся каждые 15 шагов (можно изменить параметром --log_interval).

5. После завершения всех режимов скопируйте папки с результатами на локальную машину:
   scp -r root@<IP_сервера>:/root/results_* /локальный/путь/

6. Выключите сервер, чтобы не тратить бюджет.

Вычисляет метрики: ROUGE-L, BERTScore, F1 на фактах (числа, даты).
Результаты сохраняются в CSV и JSON.

Новое: промежуточный вывод каждые 15 шагов, чтобы контролировать процесс.
"""

import os
import sys
import json
import csv
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Set

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset
import evaluate
from tqdm import tqdm

# Глобальный логгер модуля
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Оценка модели с/без RAG")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["base", "base_rag", "finetuned_rag"],
                        help="Режим оценки")
    parser.add_argument("--model_name", type=str,
                        default="Vikhrmodels/QVikhr-3-4B-Instruction",
                        help="Базовая модель")
    parser.add_argument("--adapter_path", type=str,
                        help="Путь к адаптерам LoRA (только для finetuned_rag)")
    parser.add_argument("--test_file", type=str,
                        default="data/splits/test.jsonl",
                        help="Путь к test.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="evaluation_results",
                        help="Директория для сохранения результатов")
    parser.add_argument("--log_dir", type=str,
                        default="./logs",
                        help="Директория для сохранения логов")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Максимальное число новых токенов")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Максимальная длина входной последовательности (в токенах)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Устройство для инференса")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Использовать 4-битное квантование (только для CUDA)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"],
                        help="Тип данных для вычислений (при 4bit используется bnb_4bit_compute_dtype)")
    parser.add_argument("--target_rouge", type=float, default=0.50,
                        help="Целевое значение ROUGE-L")
    parser.add_argument("--target_bert", type=float, default=0.80,
                        help="Целевое значение BERTScore F1")
    parser.add_argument("--target_facts", type=float, default=0.75,
                        help="Целевое значение F1 на фактах")
    parser.add_argument("--sampling", action="store_true", default=False,
                        help="Использовать сэмплирование (иначе жадный поиск)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Температура (если sampling=True)")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Включить отладочный вывод")
    parser.add_argument("--log_interval", type=int, default=15,
                        help="Выводить промежуточные результаты каждые N шагов")
    return parser.parse_args()


def setup_logging(log_dir: str):
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_filename = f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir_path / log_filename

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_args(args):
    if args.mode == "finetuned_rag" and not args.adapter_path:
        raise ValueError("Для режима finetuned_rag необходимо указать --adapter_path")

    if args.sampling and args.temperature <= 0:
        logger.warning("Температура <= 0 при включённом сэмплировании, устанавливаю 0.1")
        args.temperature = 0.1

    test_path = Path(args.test_file)
    if not test_path.is_file():
        raise FileNotFoundError(f"Тестовый файл не найден: {test_path}")

    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
        if not adapter_path.is_dir():
            raise NotADirectoryError(f"Путь к адаптерам не существует или не директория: {adapter_path}")


def setup_model_tokenizer(
    model_name: str,
    adapter_path: Optional[str] = None,
    device: str = "cuda",
    use_4bit: bool = True,
    dtype_str: str = "float16"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info(f"Загрузка модели: {model_name}")
    if adapter_path:
        logger.info(f"Применение адаптеров из: {adapter_path}")

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA не доступна, переключаюсь на CPU")
        device = "cpu"

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype_str, torch.float16)

    quantization_config = None
    if use_4bit:
        if device != "cuda":
            logger.warning("4-битное квантование требует CUDA, отключаю")
            use_4bit = False
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype
            )
            logger.info("4-битное квантование включено")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if not hasattr(tokenizer, 'apply_chat_template'):
        raise AttributeError("Токенизатор не поддерживает apply_chat_template. Обновите transformers или выберите другую модель.")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("pad_token установлен в eos_token")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Добавлен новый pad_token: [PAD]")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config
    )

    if tokenizer.pad_token == '[PAD]':
        if hasattr(model, 'resize_token_embeddings'):
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info("Эмбеддинги расширены и config.pad_token_id обновлён")
        else:
            logger.warning("Модель не поддерживает resize_token_embeddings, возможно, проблемы с pad_token")

    if adapter_path:
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Адаптеры не найдены: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        logger.info("Адаптеры успешно применены")

    model.eval()
    model.requires_grad_(False)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Всего параметров: {total_params:,}")
    logger.info(f"Устройство модели: {next(model.parameters()).device}")

    return model, tokenizer


def load_test_data(test_file: Path) -> List[Dict[str, Any]]:
    if not test_file.is_file():
        raise FileNotFoundError(f"Тестовый файл не найден: {test_file}")

    dataset = load_dataset('json', data_files=str(test_file), split='train')
    required_fields = ['doc_id', 'question', 'answer', 'context']
    missing = [f for f in required_fields if f not in dataset.column_names]
    if missing:
        raise ValueError(f"В датасете отсутствуют поля: {missing}. Убедитесь, что используется правильный test.jsonl (с полем context).")

    data = [dataset[i] for i in range(len(dataset))]
    logger.info(f"Загружено {len(data)} тестовых примеров")
    return data


def build_prompt(tokenizer, question: str, context: Optional[str] = None) -> str:
    if context:
        user_content = f"Контекст: {context}\nВопрос: {question}"
    else:
        user_content = f"Вопрос: {question}"
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def truncate_context(tokenizer, question: str, context: str, max_length: int, max_new_tokens: int) -> str:
    if not context:
        return ""
    prompt_without_context = build_prompt(tokenizer, question, None)
    base_len = len(tokenizer.encode(prompt_without_context))
    available = max_length - max_new_tokens - base_len
    if available <= 0:
        logger.warning(f"Нет места для контекста (available={available}), контекст будет пустым")
        return ""
    context_tokens = tokenizer.encode(context, truncation=True, max_length=available)
    return tokenizer.decode(context_tokens, skip_special_tokens=True)


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int, max_length: int,
                    sampling: bool = False, temperature: float = 0.1) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        if sampling:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
    input_len = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][input_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not answer:
        logger.warning("Модель вернула пустой ответ")
    return answer


def extract_facts(text: str) -> Set[str]:
    facts = set()
    number_pattern = r'\b\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?\b'
    for match in re.finditer(number_pattern, text):
        num_str = match.group().replace(',', '.')
        try:
            num_float = float(num_str)
            if num_float.is_integer():
                norm = str(int(num_float))
            else:
                norm = str(num_float).rstrip('0').rstrip('.') if '.' in str(num_float) else str(num_float)
            facts.add(norm)
        except ValueError:
            continue

    date_pattern_dot = r'\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b'
    for match in re.finditer(date_pattern_dot, text):
        d, m, y = match.groups()
        d = d.zfill(2); m = m.zfill(2)
        if len(y) == 2:
            y = f"20{y}"
        facts.add(f"{d}.{m}.{y}")

    date_pattern_hyphen = r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b'
    for match in re.finditer(date_pattern_hyphen, text):
        y, m, d = match.groups()
        d = d.zfill(2); m = m.zfill(2)
        facts.add(f"{d}.{m}.{y}")

    return facts


def compute_f1_facts(generated: str, reference: str) -> Tuple[float, bool, bool]:
    gen_facts = extract_facts(generated)
    ref_facts = extract_facts(reference)
    has_ref = len(ref_facts) > 0
    has_gen = len(gen_facts) > 0

    if not has_ref:
        return (1.0 if not has_gen else 0.0, has_ref, has_gen)
    if not has_gen:
        return (0.0, has_ref, has_gen)
    intersection = gen_facts.intersection(ref_facts)
    precision = len(intersection) / len(gen_facts)
    recall = len(intersection) / len(ref_facts)
    if precision + recall == 0:
        return (0.0, has_ref, has_gen)
    f1 = 2 * precision * recall / (precision + recall)
    return (f1, has_ref, has_gen)


def evaluate_example(model, tokenizer, item: Dict[str, Any], args,
                     rouge_metric, bertscore_metric) -> Dict[str, Any]:
    question = item['question']
    reference = item['answer']
    context = item.get('context', None)

    if args.mode in ("base_rag", "finetuned_rag"):
        if context is None:
            raise ValueError(f"Отсутствует поле context в примере {item['doc_id']}")
        if args.debug:
            logger.info(f"Исходный контекст (первые 200): {context[:200]}")
        context = truncate_context(tokenizer, question, context, args.max_length, args.max_new_tokens)
        if args.debug and context:
            logger.info(f"Обрезанный контекст (первые 200): {context[:200]}")
    else:
        context = None

    prompt = build_prompt(tokenizer, question, context)
    generated = generate_answer(model, tokenizer, prompt, args.max_new_tokens, args.max_length,
                                sampling=args.sampling, temperature=args.temperature)

    rouge_result = rouge_metric.compute(predictions=[generated], references=[reference])
    rouge_l = rouge_result['rougeL']
    bert_result = bertscore_metric.compute(predictions=[generated], references=[reference], lang="ru")
    bert_f1 = bert_result['f1'][0]
    f1_facts, has_ref_facts, has_gen_facts = compute_f1_facts(generated, reference)

    return {
        'doc_id': item['doc_id'],
        'question': question,
        'reference': reference,
        'generated': generated,
        'rougeL': rouge_l,
        'bertscore_f1': bert_f1,
        'f1_facts': f1_facts,
        'has_ref_facts': has_ref_facts,
        'has_gen_facts': has_gen_facts
    }


def main():
    args = parse_args()
    setup_logging(args.log_dir)

    logger.info("=" * 60)
    logger.info("ЗАПУСК ОЦЕНКИ МОДЕЛИ")
    logger.info("=" * 60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    try:
        validate_args(args)
    except Exception as e:
        logger.error(f"Ошибка в аргументах: {e}")
        sys.exit(1)

    logger.info(f"Режим: {args.mode}")
    logger.info(f"Тестовый файл: {args.test_file}")
    logger.info(f"Выходная директория: {args.output_dir}")
    logger.info(f"Директория логов: {args.log_dir}")
    logger.info(f"Промежуточный вывод каждые {args.log_interval} шагов")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        test_data = load_test_data(Path(args.test_file))
    except Exception as e:
        logger.exception(f"Ошибка загрузки тестовых данных: {e}")
        sys.exit(1)

    try:
        model, tokenizer = setup_model_tokenizer(
            model_name=args.model_name,
            adapter_path=args.adapter_path if args.mode == "finetuned_rag" else None,
            device=args.device,
            use_4bit=args.use_4bit,
            dtype_str=args.dtype
        )
    except Exception as e:
        logger.exception(f"Ошибка загрузки модели: {e}")
        sys.exit(1)

    logger.info("Загрузка метрик...")
    rouge_metric = evaluate.load('rouge')
    bertscore_metric = evaluate.load('bertscore')

    results = []
    running_rouge = 0.0
    running_bert = 0.0
    running_f1 = 0.0
    count = 0

    for idx, item in enumerate(tqdm(test_data, desc=f"Оценка {args.mode}")):
        try:
            res = evaluate_example(model, tokenizer, item, args, rouge_metric, bertscore_metric)
            results.append(res)
            running_rouge += res['rougeL']
            running_bert += res['bertscore_f1']
            running_f1 += res['f1_facts']
            count += 1

            if count % args.log_interval == 0:
                avg_rouge = running_rouge / count
                avg_bert = running_bert / count
                avg_f1 = running_f1 / count
                logger.info(f"=== Промежуточные результаты после {count} примеров ===")
                logger.info(f"  ROUGE-L: {avg_rouge:.4f} (цель ≥{args.target_rouge})")
                logger.info(f"  BERTScore F1: {avg_bert:.4f} (цель ≥{args.target_bert})")
                logger.info(f"  F1 факты: {avg_f1:.4f} (цель ≥{args.target_facts})")
                if args.debug:
                    # Покажем один пример из последних
                    logger.info(f"Пример {count}:")
                    logger.info(f"  Вопрос: {item['question'][:100]}...")
                    logger.info(f"  Эталон: {item['answer'][:100]}...")
                    logger.info(f"  Ответ: {res['generated'][:100]}...")
                    if args.mode in ("base_rag", "finetuned_rag"):
                        ctx = item.get('context', '')
                        logger.info(f"  Контекст (первые 200): {ctx[:200]}...")

        except Exception as e:
            logger.error(f"Ошибка при обработке примера {item.get('doc_id', 'unknown')}: {e}")
            continue

    if not results:
        logger.error("Нет успешно обработанных примеров.")
        sys.exit(1)

    # Сохраняем полные результаты
    csv_path = output_dir / f"results_{args.mode}.csv"
    logger.info(f"Сохранение CSV в {csv_path}")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['doc_id', 'question', 'reference', 'generated',
                                                'rougeL', 'bertscore_f1', 'f1_facts', 'has_ref_facts', 'has_gen_facts'])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    avg_rouge = float(running_rouge / count)
    avg_bert = float(running_bert / count)
    avg_f1 = float(running_f1 / count)
    examples_with_facts = int(sum(1 for r in results if r['has_ref_facts']))
    examples_without_facts = len(results) - examples_with_facts

    summary = {
        'mode': args.mode,
        'num_examples': len(results),
        'examples_with_facts': examples_with_facts,
        'examples_without_facts': examples_without_facts,
        'avg_rougeL': avg_rouge,
        'avg_bertscore_f1': avg_bert,
        'avg_f1_facts': avg_f1,
        'target_rougeL': args.target_rouge,
        'target_bertscore_f1': args.target_bert,
        'target_f1_facts': args.target_facts,
        'rougeL_met': bool(avg_rouge >= args.target_rouge),
        'bertscore_met': bool(avg_bert >= args.target_bert),
        'f1_facts_met': bool(avg_f1 >= args.target_facts),
    }

    summary_path = output_dir / f"summary_{args.mode}.json"
    logger.info(f"Сохранение JSON в {summary_path}")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Результаты сохранены в {output_dir}")
    logger.info(f"ИТОГОВЫЕ МЕТРИКИ для {args.mode}:")
    logger.info(f"  ROUGE-L: {avg_rouge:.4f} (цель ≥{args.target_rouge}) {'✅' if avg_rouge>=args.target_rouge else '❌'}")
    logger.info(f"  BERTScore F1: {avg_bert:.4f} (цель ≥{args.target_bert}) {'✅' if avg_bert>=args.target_bert else '❌'}")
    logger.info(f"  F1 факты: {avg_f1:.4f} (цель ≥{args.target_facts}) {'✅' if avg_f1>=args.target_facts else '❌'}")
    logger.info(f"  Примеров с фактами: {examples_with_facts}, без фактов: {examples_without_facts}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
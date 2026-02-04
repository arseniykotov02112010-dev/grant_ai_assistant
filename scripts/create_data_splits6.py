import json
import random
from pathlib import Path
from collections import defaultdict


def create_splits(dataset_path: Path, output_dir: Path, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Разделяет датасет на train/val/test по документам.
    Сохраняет в output_dir файлы train.jsonl, val.jsonl, test.jsonl.
    """
    random.seed(seed)

    # 1. Загрузка и группировка данных по doc_id
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]

    grouped_by_doc = defaultdict(list)
    for item in lines:
        grouped_by_doc[item['doc_id']].append(item)

    unique_docs = list(grouped_by_doc.keys())
    random.shuffle(unique_docs)

    # 2. Расчет границ разделения
    total_docs = len(unique_docs)
    train_end = int(total_docs * train_ratio)
    val_end = train_end + int(total_docs * val_ratio)

    train_docs = unique_docs[:train_end]
    val_docs = unique_docs[train_end:val_end]
    test_docs = unique_docs[val_end:]

    # 3. Запись сплитов
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        'train': train_docs,
        'val': val_docs,
        'test': test_docs
    }

    stats = {}
    for split_name, doc_list in splits.items():
        output_path = output_dir / f'{split_name}.jsonl'
        with open(output_path, 'w', encoding='utf-8') as f_out:
            count_items = 0
            for doc_id in doc_list:
                for item in grouped_by_doc[doc_id]:
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    count_items += 1
        stats[split_name] = {'docs': len(doc_list), 'qa_pairs': count_items}

    # 4. Вывод статистики
    print("=" * 60)
    print("РАЗДЕЛЕНИЕ ДАННЫХ ЗАВЕРШЕНО")
    print("=" * 60)
    for split, numbers in stats.items():
        print(f"{split.upper():10} | Документов: {numbers['docs']:4} | QA-пар: {numbers['qa_pairs']:5}")
    print(f"{'Всего':10} | Документов: {total_docs:4} | QA-пар: {len(lines):5}")
    print(f"\nФайлы сохранены в: {output_dir.absolute()}")

    # Сохранение статистики для отчета
    with open(output_dir / 'split_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    # ========== НАСТРОЙКИ ==========
    BASE_DIR = Path(__file__).parent.parent  # Корень проекта
    DATASET_PATH = BASE_DIR / "data" / "processed" / "sft_dataset.jsonl"
    OUTPUT_DIR = BASE_DIR / "data" / "splits"
    # ===============================

    if not DATASET_PATH.exists():
        print(f"Ошибка: файл с датасетом не найден: {DATASET_PATH}")
        exit(1)

    stats = create_splits(DATASET_PATH, OUTPUT_DIR)
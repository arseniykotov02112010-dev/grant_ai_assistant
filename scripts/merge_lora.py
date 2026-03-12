import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Пути
base_model_name = "Vikhrmodels/QVikhr-3-4B-Instruction"
adapter_path = "model/qvikhr-grant-lora"        # папка с адаптерами
output_path = "model/qvikhr-grant-merged"       # куда сохранить результат

# Загрузка базовой модели на CPU (без device_map, чтобы избежать offload)
print("Загрузка базовой модели...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="cpu",               # явно загружаем на CPU
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

# Загрузка адаптеров LoRA
print("Загрузка адаптеров LoRA...")
model = PeftModel.from_pretrained(model, adapter_path)

# Слияние
print("Слияние адаптеров с базовой моделью...")
model = model.merge_and_unload()

# Сохранение объединённой модели
print(f"Сохранение в {output_path}...")
os.makedirs(output_path, exist_ok=True)
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("Готово! Модель сохранена.")
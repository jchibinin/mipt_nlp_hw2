import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import gradio as gr
import kagglehub
import glob
from peft import get_peft_model, LoraConfig, TaskType

# Подготовка данных
def prepare_dialogue_data(all_seasons):
    dialogues = []
    current_dialogue = []
    
    for i in range(len(all_seasons)):
        row = all_seasons.iloc[i]
        
        # Добавляем реплику в текущий диалог
        current_dialogue.append(f"{row['name']}: {row['line']}")
        
        # Если следующая строка отсутствует или между репликами большой промежуток
        # (предполагаем, что это новая сцена), сохраняем текущий диалог
        if (i == len(all_seasons) - 1 or  # последняя строка
            all_seasons.iloc[i + 1]['name'] == all_seasons.iloc[i]['name'] or  # тот же персонаж говорит дважды
            len(current_dialogue) >= 10):  # достигнут максимальный размер диалога
            
            if len(current_dialogue) > 1:  # Сохраняем только диалоги с более чем одной репликой
                dialogues.append(" | ".join(current_dialogue))
            current_dialogue = []
    
    return dialogues

# Создаем датасет
class HouseDialogueDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_length=512):
        self.encodings = tokenizer(
            dialogues,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

# Функция для обучения модели
def train_house_model(model, train_dataloader, device, num_epochs=3):
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-4)  # Немного увеличим learning rate
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,  # Добавим warmup steps
        num_training_steps=len(train_dataloader) * num_epochs
    )
    
    # Включаем вывод градиентов для отслеживания обучения
    model.print_trainable_parameters()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Average loss: {avg_loss}")
    
    return model

# Функция генерации ответов
def generate_house_response(prompt, model, tokenizer, max_length=100):
    model.eval()
    
    # Подготовка входного текста с attention mask
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Перемещаем входные данные на то же устройство, что и модель
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Извлекаем только ответ Хауса
    try:
        house_response = response.split("House:")[-1].strip()
    except:
        house_response = response
    
    return house_response

# Загрузка данных из kaggle
def load_house_data():
    path = kagglehub.dataset_download("kunalbhar/house-md-transcripts")
    
    seasons = []
    for file in glob.glob(path+'\\season*.csv'):
        df = pd.read_csv(file, encoding='unicode_escape')
        seasons.append(df)
    
    all_seasons = pd.concat(seasons)
    return all_seasons

# Основной код для обучения
def main():
    try:
        print("Пытаемся загрузить существующую модель...")
        base_model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # LoRA conf
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # ранг матрицы адаптации
            lora_alpha=32,  # параметр масштабирования
            lora_dropout=0.1,  # dropout для LoRA слоев
            target_modules=["c_attn", "c_proj"]  # целевые слои для настройки
        )
        
        # PEFT модель
        model = get_peft_model(base_model, peft_config)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Проверяем наличие сохраненных весов LoRA
        try:
            model.load_adapter("house_model_lora")
            print("Загружены сохраненные веса LoRA")
        except:
            print("Начинаем обучение с LoRA...")
            
            # Обучаем модель
            all_seasons = load_house_data()
            dialogues = prepare_dialogue_data(all_seasons)
            
            dataset = HouseDialogueDataset(dialogues, tokenizer)
            train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            model = train_house_model(model, train_dataloader, device)
            
            # cохраняем только веса LoRA
            model.save_pretrained("house_model_lora")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        raise e

def create_chat_interface(model, tokenizer):
    def chat_with_house(message, history):
        
        context = ""
        if history:
            context = " | ".join([f"User: {q}\nHouse: {a}" for q, a in history[-3:]])
            prompt = f"{context} | User: {message}\nHouse:"
        else:
            prompt = f"User: {message}\nHouse:"
        
        response = generate_house_response(prompt, model, tokenizer)
        return response

    demo = gr.ChatInterface(
        fn=chat_with_house,
        title="Чат с доктором Хаусом (v2)",
        
    )
    
    return demo

# Запуск обучения и создание интерфейса
if __name__ == "__main__":
    model, tokenizer = main()
    demo = create_chat_interface(model, tokenizer)
    demo.launch()
import os
import json
import random
from tqdm import tqdm
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model, TaskType
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_json_field(folder, field=None):
    data = []
    all_files = list(Path(folder).glob("*.json"))
    for file in all_files:
        with open(file, "r", encoding="utf-8") as f:
            content = json.load(f)
            if field:
                if isinstance(content, list):
                    data.extend([item[field] for item in content if field in item])
                elif isinstance(content, dict) and field in content:
                    data.append(content[field])
            else:
                if isinstance(content, list):
                    data.extend(content)
                elif isinstance(content, str):
                    data.append(content)
    return data

def combine_template_instruction(template, instruction):
    return f"{template.strip()}\n\n{instruction.strip()}"

def create_causal_data(text, tokenizer, max_length=512):
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

class PromptRandomSampleDataset(Dataset):
    def __init__(self, templates, instructions, tokenizer, total_samples):
        self.prompts = [combine_template_instruction(t, i) for t in templates for i in instructions]
        self.prompts = self.prompts[::1]
        self.tokenizer = tokenizer
        self.total_samples = len(self.prompts)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        return create_causal_data(prompt, self.tokenizer)

template_dir = ""
prompt_dir = ""
save_path = ""
os.makedirs(save_path, exist_ok=True)

model_path = ""
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)

templates = load_json_field(template_dir)
instructions = load_json_field(prompt_dir, "instruction")

target_sample_count = len(templates) * len(instructions)

dataset = PromptRandomSampleDataset(templates, instructions, tokenizer, total_samples=target_sample_count)

all_indices = list(range(len(dataset)))
random.shuffle(all_indices)
subset_indices = all_indices[:len(all_indices) // 1]
dataset = Subset(dataset, subset_indices)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=0.0001)
num_epochs = 10
total_steps = len(dataloader) * num_epochs

model.train()

for epoch in range(num_epochs):
    running_loss = 0.0

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
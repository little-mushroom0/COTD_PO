import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
    # DataCollatorForLanguageModeling
)
from datasets import Dataset, DatasetDict
from datasets import load_dataset
import wandb
import json
import numpy as np
import re

set_seed(42)

# 3. 模型与分词器初始化

MODEL_NAME = "/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

from datasets import load_dataset

# 加载数据集
train_data = load_dataset('json', data_files={'train': './train-llama-new1.jsonl'}, split='train').shuffle(seed=42)
val_data = load_dataset('json', data_files={'val': './val-llama-new1.jsonl'}, split='val').shuffle(seed=42)
# 3. 数据预处理：转换为适合模型的格式
def preprocess_data(data):
    
        
    prompt = data['prompt']
    answers = data['answer']
    rewards = data['reward']
    thought = data['thought']
    
    thought = re.sub(r'My thought:|My thought|My answer|My answer:', '', thought)

    answer_label = []
    thought_label = []
    reward_i = []
    new_input_ids = []
    new_attention_mask = []

    prompt_tem = f"<|start_header_id|>user<|end_header_id|>\n\nYour task is to answer the question below. Provide a brief, step-by-step analysis of your thought process in answering the question, and then give your answer to the question. Please use the format 'My thought: ...' and 'My answer: ...'. \nQuestion: {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    thought_tem = f"My thought: {thought}My answer:"
    prompt_tokenized = tokenizer(prompt_tem, add_special_tokens=False)
    thought_tokenized = tokenizer(thought_tem, add_special_tokens=False)

    # 对每个答案进行处理
    for answer, reward in zip(answers, rewards):
        answer = re.sub(r'My thought:|My thought|My answer|My answer:', '', answer)
        
        prompt_thought_answer = f"<|start_header_id|>user<|end_header_id|>\n\nYour task is to answer the question below. Provide a brief, step-by-step analysis of your thought process in answering the question, and then give your answer to the question. Please use the format 'My thought: ...' and 'My answer: ...'. \nQuestion: {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nMy thought: {thought}My answer: {answer}<|eot_id|>"
        full_tokenized = tokenizer(prompt_thought_answer, add_special_tokens=False)
    
        # 检查当前总的 token 长度是否超出了最大长度
        # total_length = len(full_tokenized["input_ids"])

        # if total_length > :
        #     print(f"Skipping sample due to long length: {total_length}")
        #     continue  # 如果超长，则跳过当前样本
        reward_i.append(reward)

        # prompt_tem = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        # thought_tem = f"My thought: {thought}My answer:"

        
        # prompt_tokenized = tokenizer(prompt_tem, add_special_tokens=False)
        # thought_tokenized = tokenizer(thought_tem, add_special_tokens=False)

            
        thought_ids = thought_tokenized["input_ids"]
        for j in range(len(full_tokenized["input_ids"])):
            if full_tokenized["input_ids"][j:j+len(thought_ids)] == thought_ids:
                # print("not")
                break
        # print(input_ids[j:j+len(thought_ids)], thought_ids)
        if full_tokenized["input_ids"][j:j+len(thought_ids)] == thought_ids:
            thought_labels_i = [-100]*len(full_tokenized["input_ids"])
            thought_labels_i[j:j+len(thought_ids)] = full_tokenized["input_ids"][j:j+len(thought_ids)]
            answer_labels_i = [-100]*len(full_tokenized["input_ids"])
            answer_labels_i[j+len(thought_ids):] = full_tokenized["input_ids"][j+len(thought_ids):]
            
            thought_label.append(thought_labels_i)
            answer_label.append(answer_labels_i)
            new_input_ids.append(full_tokenized["input_ids"])
            new_attention_mask.append(full_tokenized["attention_mask"])
        else:
            print("wrong")
            print(full_tokenized["input_ids"], thought_ids)

    return {
        'input_ids': new_input_ids,
        'attention_mask': new_attention_mask,
        'thought_labels': thought_label,
        'answer_labels' : answer_label,
        'rewards': reward_i
    }


train_dataset = train_data.map(preprocess_data, num_proc=4)
val_dataset = val_data.map(preprocess_data, num_proc=4)
print(train_dataset[0])

# 5. 自定义数据整理器
# 修改后的数据整理器
class PromptLevelCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        batch_input_ids = []
        batch_attention_mask = []
        batch_thought_labels = []
        batch_answer_labels = []
        batch_rewards = []

        max_answers = max(len(x["input_ids"]) for x in features)
        max_len = max(max(len(ids) for ids in x["input_ids"]) for x in features)

        for item in features:
            num_answers = len(item["input_ids"])
            # pad answers to same number (e.g., 8), then pad each sequence to same length
            padded_input_ids = []
            padded_attention_mask = []
            padded_thought_labels = []
            padded_answer_labels = []

            for i in range(num_answers):
                pad_len = max_len - len(item["input_ids"][i])
                padded_input_ids.append([self.pad_token_id]*pad_len + item["input_ids"][i])
                padded_attention_mask.append([0]*pad_len + item["attention_mask"][i])
                padded_thought_labels.append([-100]*pad_len + item["thought_labels"][i])
                padded_answer_labels.append([-100]*pad_len + item["answer_labels"][i])

            # 如果不够 max_answers，就 pad 空答案
            while len(padded_input_ids) < max_answers:
                padded_input_ids.append([self.pad_token_id]*max_len)
                padded_attention_mask.append([0]*max_len)
                padded_thought_labels.append([-100]*max_len)
                padded_answer_labels.append([-100]*max_len)

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
            batch_thought_labels.append(padded_thought_labels)
            batch_answer_labels.append(padded_answer_labels)
            batch_rewards.append(item["rewards"] + [0.0] * (max_answers - num_answers))

        return {
            "input_ids": torch.tensor(batch_input_ids),           # [B, N, L]
            "attention_mask": torch.tensor(batch_attention_mask),
            "thought_labels": torch.tensor(batch_thought_labels),
            "answer_labels": torch.tensor(batch_answer_labels),
            "rewards": torch.tensor(batch_rewards, dtype=torch.float)
        }

# 6. 自定义Trainer
class PromptLevelTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        def normalize_rewards(rewards):
            """
            对 rewards 进行最小-最大标准化，映射到 [0, 1] 区间。
            """
            min_rewards = torch.min(rewards, dim=-1, keepdim=True).values
            max_rewards = torch.max(rewards, dim=-1, keepdim=True).values
            normalized_rewards = (rewards - min_rewards)/(max_rewards-min_rewards+0.001)*5
            return normalized_rewards
        B, N, L = inputs["input_ids"].shape
        # print(inputs["input_ids"].shape)
        rewards = inputs["rewards"]  # [B, N]
        rewards = normalize_rewards(rewards)  # 归一化 reward


        answer_losses = []
        for b in range(B):
            logps = []
            for n in range(N):
              
                input_ids = inputs["input_ids"][b, n].unsqueeze(0)
                attention_mask = inputs["attention_mask"][b, n].unsqueeze(0)
                answer_labels = inputs["answer_labels"][b, n].unsqueeze(0)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = answer_labels[..., 1:].contiguous()

                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='none'
                )
                token_logp = -0.5*loss.mean()  # Negative because cross_entropy returns loss
                logps.append(token_logp)

            logps = torch.stack(logps)  # [N]
            # logps = normalize_rewards(logps)
            p_t = torch.softmax(rewards[b], dim=-1)
            p_s_log = torch.log_softmax(logps, dim=-1)
            # print("rewards",rewards)
            # print("p_t",p_t)
            # print("logps",logps)
            # print("p_s_log",p_s_log)
            answer_loss = -torch.sum(p_t * p_s_log)
            answer_losses.append(answer_loss)

        avg_answer_loss = torch.stack(answer_losses).mean()

        max_indices = torch.argmax(rewards, dim=1)  # [B]
        
        # Gather corresponding sequences
        input_ids_selected = inputs["input_ids"][torch.arange(B), max_indices, :]  # [B, L]
        attention_mask_selected = inputs["attention_mask"][torch.arange(B), max_indices, :]  # [B, L]
        labels_selected = inputs["answer_labels"][torch.arange(B), max_indices, :]  # [B, L]

        # Compute NLL loss for the entire sequence (thought + selected answer)
        outputs = model(input_ids=input_ids_selected, attention_mask=attention_mask_selected)
        logits = outputs.logits

        # # thought loss
        # input_ids_all = inputs["input_ids"].view(B, N, L)[:,0,:]
        # attention_mask_all = inputs["attention_mask"].view(B, N, L)[:, 0, :]
        thought_labels_all = inputs["thought_labels"][torch.arange(B), max_indices, :] 
        shift_logits = logits[..., :-1, :].contiguous()
        shift_thought_labels = thought_labels_all[..., 1:].contiguous().view(-1)

        thought_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_thought_labels,
            ignore_index=-100
        )
        shift_labels = labels_selected[..., 1:].contiguous()
        # print(shift_logits)
        # print(shift_labels)

        # Flatten and compute loss
        answer_loss_best = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        # print(avg_answer_loss,thought_loss)
        if self.args.logging_dir:
            wandb.log({
                "avg_answer_loss": avg_answer_loss.item(),
                "thought_loss": thought_loss.item(),
                "answer_loss_best":answer_loss_best.item(),
            })
        # self.log({
        #     "avg_answer_loss": avg_answer_loss.item(),
        #     "thoughtanswer_loss": thought_loss.item(),
        #     "answer_loss_best":answer_loss_best.item(),
        # })

        return avg_answer_loss + thought_loss

# 9. 模型加载（扩展词表）
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    use_cache=False,
    # main_input_name = "input_ids"
)

# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map=None)
# model.to('cuda:0')
# 10. 训练参数配置
training_args = TrainingArguments(
    remove_unused_columns=False,
    output_dir="./llama3-8b-15-12-re-128-local-0.5",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=128,
    num_train_epochs=5,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    logging_steps=1,
    save_strategy="steps",
    save_steps=50,
    eval_steps=100,
    report_to="wandb",
    seed=42,
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    # deepspeed="/raid_sdc/home/nlj/SimPo/accelerate_configs/dpsd.json",
)

# 11. 训练执行
wandb.init(project="llama3-mixed-loss2")


trainer = PromptLevelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset = val_dataset,
    data_collator=PromptLevelCollator(tokenizer),
    tokenizer=tokenizer,
)

trainer.train()

# 12. 模型保存
# trainer.save_model("./final_model2")
# tokenizer.save_pretrained("./final_model2")
wandb.finish()

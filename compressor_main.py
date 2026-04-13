# Please Fighting! Never Give Up!
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: compressor_main.py
# ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import time
import torch
import argparse
from datasets import load_dataset
from torch.utils.data import Dataset
from compressor_model import compressor
from peft import LoraConfig, PeftConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, logging

# qwen-checkpoints/compressor/final_checkpoint
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_path", type=str, default="qwen-2.5-1.5b-instruct", help='基础模型地址')
    parser.add_argument("--dataset_path", type=str, default='datasets/gsm8k/train.jsonl', help='数据集地址')
    parser.add_argument("--lora_path", type=str, default="", help='lora存储地址')
    parser.add_argument("--output_path", type=str, default="./qwen-checkpoints/compressor/", help='模型保存路径地址')
    parser.add_argument("--logging_path", type=str, default="./qwen-checkpoints/compressor/", help='输出地址')

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--memory_num", type=int, default=40)
    parser.add_argument("--input_dim", type=int, default=1536)
    parser.add_argument("--hidden_dim", type=int, default=3000)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--max_epoch", type=int, default=-1)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)

    parser.add_argument("--strategy", type=str, default='steps')
    parser.add_argument("--eval_freq", type=int, default=100, help='评估模型频率')
    parser.add_argument("--save_freq", type=int, default=100, help='保存模型频率')
    parser.add_argument("--save_total_limit", type=int, default=3, help='保存最新的3个检查点')
    parser.add_argument("--log_freq", type=int, default=1, help='日志记录频率')

    parser.add_argument("--learning_rate", type=float, default=5e-5, help='学习率')
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help='学习率变化方式')
    parser.add_argument("--num_warmup_steps", type=int, default=300, help='学习率变化节点')
    parser.add_argument("--weight_decay", type=float, default=0.05, help='学习率变化率')
    parser.add_argument("--warmup_ratio", type=float, default=0., help='学习率变化节点另一种设置方式')

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=False, help='更快的计算速度和更低的内存使用量')
    parser.add_argument("--bf16", action="store_false", default=True, help='比FP16更快的训练和更宽的范围')
    parser.add_argument("--device", type=str, default=torch.device("cuda:0"))
    parser.add_argument("--seed", type=int, default=1103)

    parser.add_argument("--lora_r", type=int, default=16, help='LoRA中的秩')
    parser.add_argument("--lora_alpha", type=int, default=32, help='LoRA超参数，用于缩放低秩适应的权重')
    parser.add_argument("--lora_dropout", type=float, default=0.05, help='LoRA层的丢弃率')
    parser.add_argument("--run_name", type=str, default="qwen-compress", help='进程名')

    parser.add_argument("--do_eval", action="store_false", default=True, help='是否评估')
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help='断点重训')
    parser.add_argument("--ignore_pad_token_for_loss", action="store_false", default=True, help='忽略答案外的损失')
    parser.add_argument("--lora_target_modules", type=str, default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], help='LoRA添加位置')

    return parser.parse_args()


def read_jsonl_file(file_path):
    dataset = load_dataset("json", data_files=file_path, split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=1103)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    return train_data, valid_data


class SFTDataset(Dataset):
    def __init__(
            self,
            dataset,
            args
    ):
        self.data = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, use_fast=True)
        soft_prompt_token = "<soft_prompt>"
        self.tokenizer.add_tokens([soft_prompt_token])
        self.memory_num = args.memory_num
        self.soft_prompt_token_ids = self.tokenizer.convert_tokens_to_ids(soft_prompt_token)
        self.soft_prompt_ids = [self.soft_prompt_token_ids] * self.memory_num
        self.max_compress_length = args.max_length
        self.max_recovery_length = args.max_length
        self.ignore_pad_token_for_loss = args.ignore_pad_token_for_loss

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        cot = self.data[idx]['cot']

        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease summarize the "
                  "following reasoning process.\n") + f"{cot}<|im_end|>\n<|im_start|>assistant\n"

        input_c = self.tokenizer.encode(prompt, add_special_tokens=False, max_length=self.max_compress_length,
                                      pad_to_max_length=False, truncation=True)
        input_r = self.tokenizer.encode(cot, add_special_tokens=False, max_length=self.max_recovery_length,
                                      pad_to_max_length=False, truncation=True)

        input_ids_c = input_c[:self.max_compress_length]
        pad_len = self.max_compress_length - len(input_ids_c)
        input_ids_c = input_ids_c + self.soft_prompt_ids
        input_ids_c = [self.tokenizer.pad_token_id] * pad_len + input_ids_c
        attention_mask_c = input_ids_c
        attention_mask_c = [(1 if item != self.tokenizer.pad_token_id else 0) for item in attention_mask_c]

        input_ids_r = input_r[:self.max_recovery_length]
        pad_len = self.max_recovery_length - len(input_ids_r)
        input_ids_r = self.soft_prompt_ids + input_ids_r
        input_ids_r = input_ids_r + [self.tokenizer.pad_token_id] * pad_len
        attention_mask_r = input_ids_r
        attention_mask_r = [(1 if item != self.tokenizer.pad_token_id else 0) for item in attention_mask_r]

        labels = input_r[:self.max_recovery_length]
        labels = [self.tokenizer.pad_token_id] * (len(self.soft_prompt_ids)) + labels
        labels = labels + [self.tokenizer.pad_token_id] * pad_len

        if self.ignore_pad_token_for_loss:
            labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        return {"input_ids_c": input_ids_c,
                "attention_mask_c": attention_mask_c,
                "input_ids_r": input_ids_r,
                "attention_mask_r": attention_mask_r,
                "labels": labels}
class EvalDataset(Dataset):
    def __init__(
            self,
            dataset,
            args
    ):
        self.data = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, use_fast=True)
        soft_prompt_token = "<soft_prompt>"
        self.tokenizer.add_tokens([soft_prompt_token])
        self.memory_num = args.memory_num
        self.soft_prompt_token_ids = self.tokenizer.convert_tokens_to_ids(soft_prompt_token)
        self.soft_prompt_ids = [self.soft_prompt_token_ids] * self.memory_num
        self.max_compress_length = args.max_length
        self.max_recovery_length = args.max_length
        self.ignore_pad_token_for_loss = args.ignore_pad_token_for_loss

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        cot = self.data[idx]['cot']
        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease summarize the "
                  "following reasoning process.\n") + f"{cot}<|im_end|>\n<|im_start|>assistant\n"

        input_c = self.tokenizer.encode(prompt, add_special_tokens=False, max_length=self.max_compress_length,
                                      pad_to_max_length=False, truncation=True)
        input_r = self.tokenizer.encode(cot, add_special_tokens=False, max_length=self.max_recovery_length,
                                      pad_to_max_length=False, truncation=True)

        input_ids_c = input_c[:self.max_compress_length]
        pad_len = self.max_compress_length - len(input_ids_c)
        input_ids_c = input_ids_c + self.soft_prompt_ids
        input_ids_c = [self.tokenizer.pad_token_id] * pad_len + input_ids_c
        attention_mask_c = input_ids_c
        attention_mask_c = [(1 if item != self.tokenizer.pad_token_id else 0) for item in attention_mask_c]

        input_ids_r = self.soft_prompt_ids
        attention_mask_r = input_ids_r
        attention_mask_r = [(1 if item != self.tokenizer.pad_token_id else 0) for item in attention_mask_r]

        return {"input_ids_c": input_ids_c,
                "attention_mask_c": attention_mask_c,
                "input_ids_r": input_ids_r,
                "attention_mask_r": attention_mask_r,
                "cot": cot}

def predict_CoT(args, eval, train):
    print("Loading the model")
    if args.lora_path != "":
        lora_config = PeftConfig.from_pretrained(args.lora_path)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    print("Loading qwen + lora")

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, use_fast=True)
    soft_prompt_token = "<soft_prompt>"
    tokenizer.add_tokens([soft_prompt_token])
    model = compressor(qwen_path=args.qwen_path,
                       lora_path=args.lora_path,
                       lora_config=lora_config,
                       memory_num=args.memory_num,
                       input_dim=args.input_dim,
                       hidden_dim=args.hidden_dim)
    model = model.to(torch.bfloat16)
    eval_dataset = EvalDataset(train, args)
    batch_size = 8
    sft_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    stop_id_sequences = []
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.eos_token_id is not None:
        stop_id_sequences = [[tokenizer.eos_token_id]]
    for batch in sft_dataloader:
        input_ids_c = batch['input_ids_c']
        attention_mask_c = batch['attention_mask_c']
        input_ids_r = batch['input_ids_r']
        attention_mask_r = batch['attention_mask_r']
        generations, prompt, finish_completion = model.generate_completions(
            tokenizer=tokenizer,
            input_ids_c=input_ids_c,
            attention_mask_c=attention_mask_c,
            input_ids_r=input_ids_r,
            attention_mask_r=attention_mask_r,
            max_new_tokens=512,
            do_sample=False,
            temperature=0,
            top_p=1.0,
            stop_id_sequences=stop_id_sequences if stop_id_sequences else None,
            end_of_generation_id_sequence=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
        )
        print(batch['cot'][0])
        print(prompt[0])
        print(generations[0])
        break




def run_training(args, train_dataset, val_dataset):
    print("Loading the model")
    if args.lora_path != "":
        lora_config_name = os.path.join(
            args.output_path,
            "final_checkpoint/adapter_config.json"
        )
        lora_config = PeftConfig.from_pretrained(lora_config_name.replace('/adapter_config.json', ""))
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
    print("Loading qwen + lora")
    model = compressor(qwen_path=args.qwen_path,
                       lora_path=args.lora_path,
                       lora_config=lora_config,
                       memory_num=args.memory_num,
                       input_dim=args.input_dim,
                       hidden_dim=args.hidden_dim)
    model = model.to(torch.bfloat16)
    print(f"Total parameters of qwen: {sum(p.numel() for p in model.parameters())}")
    print("Number of trainable parameters in the model: ",
          sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"Trainable parameter ratio: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters())}")
    print("Model is on CUDA device:", torch.cuda.current_device())
    model.config = model.qwen.config
    print("model.qwen.config: ", model.qwen.config)
    print("qwen + lora loaded successfully.")

    print("Training Setting.")
    torch.autograd.set_detect_anomaly(True)
    training_args = TrainingArguments(
        do_eval=args.do_eval,
        output_dir=args.output_path,
        max_steps=args.max_steps,
        num_train_epochs=args.max_epoch,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        evaluation_strategy=args.strategy,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        warmup_steps=args.num_warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.log_freq,
        save_total_limit=args.save_total_limit,
        logging_dir=args.logging_path,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        run_name=args.run_name,
        report_to="wandb",
        save_strategy="no",
        overwrite_output_dir=False,
        dataloader_drop_last=False,
        ddp_find_unused_parameters=False if int(os.environ.get("WORLD_SIZE", 1)) != 1 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("Training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Saving last checkpoint of the model")
    final_model_path = os.path.join(args.output_path, "final_checkpoint/")
    trainer.model.lora_save(final_model_path)

    evaluation_results = trainer.evaluate()
    print("evaluation_results: ", evaluation_results)


def main(args):
    train, eval = read_jsonl_file(args.dataset_path)
    print("数据处理中.......")
    train_dataset = SFTDataset(train, args)
    print("训练数据准备完成")
    eval_dataset = SFTDataset(eval, args)
    print("验证数据准备完成")
    print("开始训练")
    start_time = time.time()
    # predict_CoT(args, eval, train)
    run_training(args, train_dataset, eval_dataset)
    end_time = time.time()
    print("训练结束")
    print("耗时：", end_time - start_time)


if __name__ == "__main__":
    args = get_args()
    assert args.qwen_path != "", "Please provide the qwen model path"
    set_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    logging.set_verbosity_error()
    main(args)
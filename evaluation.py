# Please Fighting! Never Give Up!
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: evaluation.py
# ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from time import time
from pathlib import Path
from copy import deepcopy
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from eval.utils import generate_completions
from eval.eval_script import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="outputs", help="default to `model_path`_predictions")
    parser.add_argument("--projector-path", type=str, default="qwen-checkpoints/projector/final_checkpoint")
    parser.add_argument("--lora_path", type=str, default="qwen-checkpoints/compressor/final_checkpoint")
    parser.add_argument("--compressor_path", type=str, default="qwen-2.5-1.5b-instruct")
    parser.add_argument("--answer_path", type=str, default="/data/qwen2.5/qwen-2.5-7b-instruct")
    parser.add_argument("--projector_path", type=str, default="qwen-checkpoints/projector/final_checkpoint\projector_network.pth")
    parser.add_argument("--answer_type", type=str, choices=['llama3', 'qwen'], default="qwen")
    parser.add_argument("--benchmark", type=str, choices=['gsm8k', 'math'], default="gsm8k")
    parser.add_argument("--data_type", type=str, choices=['train', 'test'], default="test")

    parser.add_argument("--input_dim", type=int, default=1536)
    parser.add_argument("--hidden_dim", type=int, default=3000)
    parser.add_argument("--output_dim", type=int, default=1536)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--memory_num", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16, help="batch size for evaluation.")
    parser.add_argument("--max_num_examples", type=int, default=16, help="maximum number of examples to evaluate.")

    return parser.parse_args()

class SelectOutput(nn.Module):
    def forward(self, x):
        return x[0]

def read_data(path):
    if path.endswith("json"):
        data = json.load(open(path, "r"))
    elif path.endswith("jsonl"):
        data = []
        with open(path, "r") as file:
            for line in file:
                line = json.loads(line)
                data.append(line)
    else:
        raise NotImplementedError()
    return data

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer(args, test_data, answer_extraction_fn):
    print("Loading model and tokenizer...")
    tokenizer_c = AutoTokenizer.from_pretrained(args.compressor_path)
    tokenizer_a = AutoTokenizer.from_pretrained(args.answer_path)
    soft_prompt_token = "<soft_prompt>"
    tokenizer_a.add_tokens([soft_prompt_token])
    tokenizer_c.add_tokens([soft_prompt_token])
    soft_prompt_token_ids = tokenizer_a.convert_tokens_to_ids(soft_prompt_token)
    prompts = []
    cots = []
    for example in test_data:
        prompt = ""
        cot = f"{example['cot']}"
        if args.answer_type == 'llama3':
            prompt += f"{tokenizer_a.bos_token}" + "<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n" + f"\n{example['question']}\n{tokenizer_a.eos_token}<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif args.answer_type == 'qwen':
            prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n" + f"\n{example['question']}<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise NotImplementedError()
        prompt = prompt.lstrip()
        example['prompt'] = prompt
        prompts.append(prompt)
        cots.append(cot)

    # set padding side to left for batch generation
    tokenizer_c.padding_side = "left"
    tokenizer_a.padding_side = "left"
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = tokenizer_a.eos_token
        tokenizer_a.pad_token_id = tokenizer_a.eos_token_id
    if tokenizer_c.pad_token is None:
        tokenizer_c.pad_token = tokenizer_c.eos_token
        tokenizer_c.pad_token_id = tokenizer_c.eos_token_id

    stop_id_sequences = []
    if tokenizer_a.eos_token_id is not None:
        stop_id_sequences = [[tokenizer_a.eos_token_id]]

    answer = AutoModelForCausalLM.from_pretrained(
        args.answer_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    answer.resize_token_embeddings(len(tokenizer_a))

    model = AutoModelForCausalLM.from_pretrained(args.compressor_path, device_map="auto")
    model.resize_token_embeddings(len(tokenizer_c))
    compressor = PeftModel.from_pretrained(model, args.lora_path)


    final_memory_embeddings_path = os.path.join(args.lora_path, "memory_embeddings.pth")
    memory_embeddings = torch.load(final_memory_embeddings_path)
    memory_embeddings = torch.nn.Parameter(memory_embeddings)

    final_memory_embeddings_path = os.path.join(args.lora_path, "compress_network.pth")
    projector_network = torch.load(final_memory_embeddings_path, weights_only=False)
    projector_network.eval()
    """for module in projector_network:
        module.to(compressor.device)"""



    start_time = time()
    outputs, finish_completion, parallel_time = generate_completions(
        memory_num=args.memory_num,
        answer=answer,
        tokenizer_a=tokenizer_a,
        compressor=compressor,
        tokenizer_c=tokenizer_c,
        memory_embeddings=memory_embeddings,
        projector_network=projector_network,
        cots=cots,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=args.temperature,
        top_p=1.0,
        batch_size=args.eval_batch_size,
        stop_id_sequences=stop_id_sequences if stop_id_sequences else None,
        end_of_generation_id_sequence=[tokenizer_a.eos_token_id] if tokenizer_a.eos_token_id is not None else None
    )
    total_time = time() - start_time
    total_time -= parallel_time
    model_outputs = outputs

    predictions = [eval(answer_extraction_fn)(item['question'], output, task='cot') for item, output in tqdm(zip(test_data, model_outputs), desc="extract answer", total=len(model_outputs))]
    assert len(model_outputs) > 0, f"{len(model_outputs)}"

    results = []
    for example, output, pred in zip(test_data, model_outputs, predictions):
        item = deepcopy(example)
        item.update({
            'model_output': output,
            'prediction': pred,
            'cot_length': 10,
            'accuracy': 0,
        })
        results.append(item)

    return results, total_time


def main(args):
    print(f"Evaluating {args.compressor_path}", flush=True)
    print(f"Max new tokens: {args.max_new_tokens}, eval batch size: {args.eval_batch_size}, temperature: {args.temperature}, seed: {args.seed}\n", flush=True)

    data_path = os.path.join('datasets', args.benchmark, f"{args.data_type}.jsonl")
    test_data = read_data(data_path)

    output_dir = os.path.join(args.output_path, "samples")
    os.makedirs(output_dir, exist_ok=True)
    print("Loading data...")

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)

    results, total_time = infer(args, test_data, answer_extraction_fn="extract_last_single_answer")

    print("Finished inference...")
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    invalid_outputs = []
    labels = []
    for item in results:
        if len(item['prediction']) == 0:
            invalid_outputs.append(
                {'prompt': item['prompt'], 'output': item['model_output'], 'answer': item['prediction']})
            res = None
        else:
            if args.benchmark=='gsm8k':
                res = eval_last_single_answer(item)
            elif args.benchmark=='math':
                res = eval_math(item)
            else:
                raise NotImplementedError()
        labels.append(res)

    for item, label in zip(results, labels):
        item['accuracy'] = label

    print("Calculating accuracy...")
    acc = 0
    num = 1
    for item in results:
        if item['accuracy'] != None:
            acc += item['accuracy']
            num += 1
    print("output acc = {:.5f}".format(acc / num * 100), flush=True)

    avg_cot_length = sum(item['cot_length'] for item in results) / len(results)
    print("output avg_cot_length = {:.5f}".format(avg_cot_length), flush=True)

    pred_fname = "predictions.jsonl"
    for item in results:
        with open(os.path.join(output_dir, pred_fname), 'a+', encoding='utf-8') as fout:
            line = json.dumps(item, ensure_ascii=False)
            fout.write(line + '\n')

    metric_fname = "metrics.json"
    with open(os.path.join(output_dir, metric_fname), "w") as fout:
        json.dump({
            "n_samples": len(results),
            "accuracy": acc / num * 100,
            "avg_cot_length": avg_cot_length,
            'sample_latency': total_time / len(test_data),
        }, fout, indent=4)





if __name__ == "__main__":
    args = get_args()
    assert args.answer_path != "", "Please provide the large language model path"
    set_seed(args.seed)
    set_random_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    main(args)
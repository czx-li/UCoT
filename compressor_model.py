# Please Fighting! Never Give Up!
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: compressor_model.py
# ---
import os
import torch
import wandb
import torch.nn as nn
from peft import get_peft_model, PeftModel
from transformers import StoppingCriteria
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

def calculate_accuracy(logits, labels):
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = labels != -100
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).float() * mask.float()
    return correct.mean().item()


class CompressNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, memory_num):
        super().__init__()
        self.memory_num = memory_num
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            for _ in range(memory_num)
        ])

    def forward(self, memory_hidden_states):
        compress_embeddings = []
        for i in range(self.memory_num):
            memory_hidden_states_i = memory_hidden_states[:, i, :]
            memory_hidden_states_i = self.layers[i](memory_hidden_states_i)
            compress_embeddings.append(memory_hidden_states_i)
        compress_embeddings = torch.stack(compress_embeddings, dim=1)
        return compress_embeddings

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences, tokenizer, prompt_length):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.tokenizer = tokenizer
        self.stop_id_sequences = stop_id_sequences
        self.stop_sequences = [tokenizer.decode(sequence) for sequence in stop_id_sequences]
        # print(f"stop sequences: {self.stop_sequences}", flush=True)
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            ids = input_ids[i][self.prompt_length:].tolist()
            should_be_stopped = False
            for stop_ids, stop_sequence in zip(self.stop_id_sequences, self.stop_sequences):
                _ids = ids
                for j in range(len(_ids), 0, -1):
                    s = self.tokenizer.decode(_ids[max(j - len(stop_ids) - 3, 0):j])
                    if s.endswith(stop_sequence):
                        should_be_stopped = True
                        break
                if should_be_stopped:
                    break
            sequences_should_be_stopped.append(should_be_stopped)
        return all(sequences_should_be_stopped)

def load_lora_parameters(model, lora_params_path, lora_config):
    """
    Initialize the LoRA parameters.

    model (AutoModelForCausalLM): LLM with LoRA parameters.
    lora_params_path (str): Pretrained LoRA parameters path for initialization.
    """
    if lora_params_path != '':
        print('Loading LoRA Parameters')
        return PeftModel.from_pretrained(model, lora_params_path)
    else:
        print('Initializing new LoRA')
        return get_peft_model(model, lora_config)

class compressor(PreTrainedModel):
    config_class = PretrainedConfig
    def __init__(self,
                 qwen_path,
                 lora_config,
                 lora_path,
                 memory_num,
                 input_dim,
                 hidden_dim,
                 config=PretrainedConfig()):
        super(compressor, self).__init__(config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, use_fast=True)
        soft_prompt_token = "<soft_prompt>"
        self.tokenizer.add_tokens([soft_prompt_token])
        model = AutoModelForCausalLM.from_pretrained(qwen_path, device_map="auto")
        model.resize_token_embeddings(len(self.tokenizer))
        self.qwen = load_lora_parameters(model, lora_path, lora_config)
        for name, param in self.qwen.named_parameters():
            param.requires_grad = False
            if 'lora' in name:
                param.requires_grad = True
        self.memory_num = memory_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if lora_path == '':
            self.memory_embeddings = nn.Parameter(
                torch.randn(1, self.memory_num, self.input_dim, dtype=torch.bfloat16).to(model.device))
            self.memory_embeddings.requires_grad = True
            self.compress_network = CompressNetwork(self.input_dim, self.hidden_dim, self.input_dim, self.memory_num)
            self.compress_network.requires_grad = True
        else:
            final_memory_embeddings_path = os.path.join(lora_path, "memory_embeddings.pth")
            final_compress_embeddings_path = os.path.join(lora_path, "compress_network.pth")
            self.memory_embeddings = torch.load(final_memory_embeddings_path)
            self.memory_embeddings = torch.nn.Parameter(self.memory_embeddings)
            self.compress_network = CompressNetwork(self.input_dim, self.hidden_dim, self.input_dim, self.memory_num)
            self.compress_network = torch.load(final_compress_embeddings_path, weights_only=False)
            self.compress_network.eval()
        self.eval_loss = 0
        self.eval_acc = 0
        self.eval_num = 0
        self.state = True


    def forward(self, input_ids_c, attention_mask_c, input_ids_r, attention_mask_r, labels):
        # CoT Compressing
        text_embeddings = self.qwen.get_input_embeddings()(input_ids_c)
        memory_embeddings = self.memory_embeddings.repeat(text_embeddings.shape[0], 1, 1).to(self.qwen.device)
        text_embeddings[:, -self.memory_num:, :] = memory_embeddings
        output = self.qwen(inputs_embeds=text_embeddings, attention_mask=attention_mask_c, output_hidden_states=True)
        hidden_states = output.hidden_states
        last_hidden_state = hidden_states[-1]
        memory_hidden_states = last_hidden_state[:, -self.memory_num:, :]
        compress_embeddings = self.compress_network(memory_hidden_states)

        # CoT Recovery
        input_embeddings = self.qwen.get_input_embeddings()(input_ids_r)
        input_embeddings[:, :self.memory_num, :] = compress_embeddings
        with self.qwen.disable_adapter():
            output = self.qwen(inputs_embeds=input_embeddings, attention_mask=attention_mask_r, labels=labels)

        logits = output.logits

        loss = output.loss

        accuracy = calculate_accuracy(logits, labels)

        if self.training and (self.state==False):
            wandb.log({'loss': loss, 'acc': accuracy})
            wandb.log({"eval_loss": (self.eval_loss / self.eval_num), "eval_acc": (self.eval_acc / self.eval_num)})
            self.eval_acc = 0
            self.eval_loss = 0
            self.eval_num = 0

        if self.training:
            self.state = True
            wandb.log({'loss': loss, 'acc': accuracy})
            return {'loss': loss, 'logits': logits}
        else:
            self.state = False
            self.eval_loss = self.eval_loss + loss
            self.eval_acc = self.eval_acc + accuracy
            self.eval_num = self.eval_num + 1
            return {'loss': loss, 'logits': logits}

    def generate_completions(self, tokenizer, input_ids_c, attention_mask_c, input_ids_r, attention_mask_r,
                             stop_id_sequences=None, end_of_generation_id_sequence=None, **generation_kwargs):
        generations = []
        prompt = []
        finish_completion = []


        if stop_id_sequences is not None:
            stop_sequences = [tokenizer.decode(stop_id_sequence) for stop_id_sequence in stop_id_sequences]

        if end_of_generation_id_sequence is not None:
            end_of_generation_sequence = tokenizer.decode(end_of_generation_id_sequence)

        num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
        generation_kwargs['use_cache'] = True

        if self.qwen.device.type == "cuda":
            input_ids_c = torch.stack(input_ids_c, dim=1).cuda()
            attention_mask_c = torch.stack(attention_mask_c, dim=1).cuda()

            input_ids_r = torch.stack(input_ids_r, dim=1).cuda()
            attention_mask_r = torch.stack(attention_mask_r, dim=1).cuda()

        batch_finish_completion = [False] * len(input_ids_c) * num_return_sequences

        # CoT Compressing
        text_embeddings = self.qwen.get_input_embeddings()(input_ids_c)
        memory_embeddings = self.memory_embeddings.repeat(text_embeddings.shape[0], 1, 1).to(self.qwen.device)
        text_embeddings[:, -self.memory_num:, :] = memory_embeddings
        output = self.qwen(inputs_embeds=text_embeddings, attention_mask=attention_mask_c, output_hidden_states=True)
        hidden_states = output.hidden_states
        last_hidden_state = hidden_states[-1]
        memory_hidden_states = last_hidden_state[:, -self.memory_num:, :]
        compress_embeddings = self.compress_network(memory_hidden_states)

        # CoT Recovery
        input_embeddings = self.qwen.get_input_embeddings()(input_ids_r)
        input_embeddings[:, :self.memory_num, :] = compress_embeddings
        with self.qwen.disable_adapter():
            batch_outputs = self.qwen.generate(inputs_embeds=input_embeddings,
                                        attention_mask=attention_mask_r,
                                        stopping_criteria=[KeyWordsCriteria(stop_id_sequences, tokenizer,
                                                input_ids_r.size(1))] if stop_id_sequences else None,
                                        pad_token_id=tokenizer.eos_token_id,
                                        **generation_kwargs)

            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(input_ids_r.shape[1], batch_outputs.shape[1]):
                        if any(tokenizer.decode(
                                batch_outputs[output_idx, token_idx: token_idx + len(stop_sequence) + 3]).startswith(
                            stop_sequence) for stop_sequence in stop_sequences):
                            if end_of_generation_id_sequence is not None and tokenizer.decode(
                                    batch_outputs[output_idx,
                                    token_idx: token_idx + len(
                                        end_of_generation_id_sequence) + 3]).startswith(end_of_generation_sequence):
                                batch_finish_completion[output_idx] = True
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(input_ids_r, skip_special_tokens=True)

        generations += batch_outputs
        prompt += batch_prompts
        finish_completion += batch_finish_completion
        return generations, prompt, finish_completion


    def lora_save(self, path):
        # 仅保存LoRA和参数网络
        self.qwen.save_pretrained(path)
        final_memory_embeddings_path = os.path.join(path, "memory_embeddings.pth")
        final_compress_embeddings_path = os.path.join(path, "compress_network.pth")
        torch.save(self.memory_embeddings, final_memory_embeddings_path)
        torch.save(self.compress_network, final_compress_embeddings_path)



# Please Fighting! Never Give Up!
# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: utils.py
# ---
import torch
import tqdm
from time import time
from transformers import StoppingCriteria


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

def generate_completions(memory_num, answer, tokenizer_a, compressor, tokenizer_c, memory_embeddings, projector_network,  cots, prompts, batch_size=1, stop_id_sequences=None,
                         end_of_generation_id_sequence=None, disable_tqdm=False, **generation_kwargs):
    generations = []
    finish_completion = []
    parallel_time = 0

    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    if stop_id_sequences is not None:
        stop_sequences = [tokenizer_a.decode(stop_id_sequence) for stop_id_sequence in stop_id_sequences]

    if end_of_generation_id_sequence is not None:
        end_of_generation_sequence = tokenizer_a.decode(end_of_generation_id_sequence)

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    generation_kwargs['use_cache'] = True


    for i in range(0, len(prompts), batch_size):

        batch_prompts = prompts[i:i + batch_size]
        batch_cots = cots[i:i + batch_size]
        tokenized_prompts = tokenizer_a(batch_prompts, padding="longest", return_tensors="pt",
                                      add_special_tokens='chatglm2' in str(answer.__class__))
        tokenized_cots = tokenizer_c(batch_cots, padding="longest", return_tensors="pt",
                                        add_special_tokens='chatglm2' in str(answer.__class__))
        thought_ids = tokenized_cots.input_ids
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        ones_to_add = torch.ones(attention_mask.size(0), memory_num, dtype=torch.long)
        attention_mask = torch.cat((attention_mask, ones_to_add), dim=1)

        if answer.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()
            thought_ids = thought_ids.cuda()

        batch_finish_completion = [False] * len(batch_prompts) * num_return_sequences
        try:
            """batch_outputs = answer.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences, tokenizer_a,
                                                    batch_input_ids.size(1))] if stop_id_sequences else None,
                pad_token_id=tokenizer_a.eos_token_id,
                **generation_kwargs
            )"""
            start_time = time()
            cot_embeddings = compressor.get_input_embeddings()(thought_ids)
            memory_embeddings = memory_embeddings.repeat(cot_embeddings.shape[0], 1, 1).to(compressor.device)
            input_embeddings = torch.cat((cot_embeddings, memory_embeddings), dim=1)
            output = compressor(inputs_embeds=input_embeddings, output_hidden_states=True)
            hidden_states = output.hidden_states
            last_hidden_state = hidden_states[-1]
            last_hidden_state = last_hidden_state.to(torch.bfloat16)
            memory_hidden_states = last_hidden_state[:, len(cot_embeddings[0]):, :]
            projector_embeddings = projector_network(memory_hidden_states)
            projector_embeddings = projector_embeddings.to(answer.device)
            total_time = time() - start_time
            parallel_time += total_time

            input_embeddings = answer.get_input_embeddings()(batch_input_ids)
            input_embeddings = torch.cat((input_embeddings, projector_embeddings), dim=1)
            input_embeddings = input_embeddings.to(torch.float16)
            batch_outputs = answer.generate(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences, tokenizer_a, batch_input_ids.size(1))] if stop_id_sequences else None,
                pad_token_id=tokenizer_a.eos_token_id,
                **generation_kwargs
            )
            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(tokenizer_a.decode(
                                batch_outputs[output_idx, token_idx: token_idx + len(stop_sequence) + 3]).startswith(stop_sequence) for stop_sequence in stop_sequences):
                            if end_of_generation_id_sequence is not None and tokenizer_a.decode(
                                    batch_outputs[output_idx,
                                    token_idx: token_idx + len(
                                        end_of_generation_id_sequence) + 3]).startswith(end_of_generation_sequence):
                                batch_finish_completion[output_idx] = True
                            batch_outputs[output_idx, token_idx:] = tokenizer_a.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer_a.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer_a.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences
        generations += batch_generations
        finish_completion += batch_finish_completion

        if not disable_tqdm:
            progress.update(len(batch_prompts) // num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations, finish_completion, parallel_time



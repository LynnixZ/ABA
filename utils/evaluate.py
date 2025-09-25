import os
from utils.prepare_data import reverse_string, remove_zero_pad
import torch
import numpy as np
from tqdm import tqdm
import random
import math
import string
import pickle
import copy
import pandas as pd
import tiktoken

from model.model import GPTConfig, GPT
from .prepare_data import *

def is_number(s):
    # handle "xey" case (e.g. 1.2e-3) - we do not use this notation in our dataset
    if 'e' in s:
        return False
    elif 'E' in s:
        return False
    elif 'inf' in s or "INF" in s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_answer_from_output(c_hat, index_hint=False):
    # Remove spaces and newlines
    c_hat_nospace = c_hat.replace(' ', '').replace('\n', '')
    # Remove content after the first '$', if any
    if '$' in c_hat_nospace:
        c_hat_nospace = c_hat_nospace.split('$')[0]
    # Extract the part after '='
    if '=' in c_hat_nospace:
        c_hat2 = c_hat_nospace.split('=')[1]
    else:
        c_hat2 = c_hat_nospace
    # Remove any non-digit characters
    if index_hint:
        import re
        c_hat2 = re.sub(r'[^0-9]', '', c_hat2)
    return c_hat2

def evaluate_addition_batch(config, model, ctx, encode, decode, verbose=False, num_digit=3, reverse_b=False, reverse_c=False,
                            algo_reason=False, operator='+', data_format='plain', verbose_correct=False, 
                            index_hint=False, zero_pad=True, zeropad_max_length=None, noorder=False, 
                            blank_space_in_equation_number=None, pad_answer=False, fix_blank_space_position=False, blank_space_exact=True,
                              max_operand_number=10, operand_number_exact=True,pad_before=False, test_limit=None, hard_mode=''):
    model.eval()
    start = config['start'] if 'start' in config.keys() else "FILE:prompt/prompt_addition_pad_test_0.01.txt"
    device = config['device']
    test_batch_size = config['test_batch_size'] if 'test_batch_size' in config.keys() else 128
    max_new_tokens = config['max_new_tokens'] if 'max_new_tokens' in config.keys() else num_digit+5
    if operator == '*': 
        max_new_tokens = num_digit*2+5
    simple= config['simple'] if 'simple' in config.keys() else False

    if blank_space_exact and blank_space_in_equation_number is not None and blank_space_in_equation_number>0: 
        max_new_tokens = blank_space_in_equation_number+5
    elif index_hint:
        max_new_tokens = 2 * num_digit + 5 

    temperature = config['temperature'] if 'temperature' in config.keys() else 0.8
    top_k = config['top_k'] if 'top_k' in config.keys() else None
    eps = config['eps'] if 'eps' in config.keys() else 0
    
    random_A= config['random_A'] if 'random_A' in config.keys() else False
    random_C= config['random_C'] if 'random_C' in config.keys() else False

    print(f'evaluating addition from: {start}')
    # print(f'max_new_tokens: {max_new_tokens}, temperature: {temperature}, top_k: {top_k}')
    if algo_reason:
        if operator == "*":
            max_new_tokens = blank_space_in_equation_number*num_digit*2+200 if ('simple' in config['dataset'] or simple) else blank_space_in_equation_number*num_digit*5+300 # TODO:
        else:
            max_new_tokens = blank_space_in_equation_number+10 
        print(f'max_new_tokens: {max_new_tokens}')
        def extract_answer_from_scratchpad(c_hat):
            c_hat = c_hat.replace(' ', '')
            # Remove spaces and newlines
            dollar_index = c_hat.find("$")
            if dollar_index == -1:
                return c_hat
                                      
            last_equal_index = c_hat.rfind("=", 0, dollar_index)
            if last_equal_index == -1:
                return c_hat
                                        
            result = c_hat[last_equal_index+1:dollar_index].strip()
            return result       
# normal addition
    test_data_file = start[5:]
    print(f"Evaluating Addition using test data file: {test_data_file}")
    # we know test examples are test.txt
    test_data_list = get_data_list(test_data_file, operator=operator)
    test_data_str, test_prompts, test_answers = generate_data_str(test_data_list, operator=operator, format=data_format, train=False, shuffle=True,  scratchpad_simple=simple,
                                        index_hint=index_hint, reverse_b=reverse_b, reverse_c=reverse_c, zero_pad=zero_pad, zeropad_max_length=zeropad_max_length,
                                        blank_space_in_equation_number=blank_space_in_equation_number,  fix_blank_space_position=fix_blank_space_position, 
                                        blank_space_number_exact=blank_space_exact, max_operand_number=max_operand_number, operand_number_exact=operand_number_exact ,pad_before=pad_before,
                                        hard_mode=hard_mode
                                            )

    lines = test_prompts
    answers = test_answers 
    total = len(lines)
    if test_limit is not None and test_limit < total:
        orig_total = total
        lines = lines[:test_limit]
        answers = answers[:test_limit]
        total = len(lines)
        print(f"[Note] test_limit={test_limit}, using first {total}/{orig_total} examples.")
    if len(answers) != total:
        print(f"[Warning] # of lines ({total}) != # of answers ({len(answers)})")

    correct = 0
    carry_dictionary = {f'carry{i}_correct': 0 for i in range(num_digit+1)}
    for i in range(num_digit+1):
        carry_dictionary[f'carry{i}_total'] = 0

    prompt_dict = {}
    for idx, prompt in enumerate(lines):
        prompt_str = prompt
        start_ids = encode(prompt_str)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        prompt_length = len(start_ids)
        input_tuple = (x, prompt_length, idx)
        if prompt_length not in prompt_dict:
            prompt_dict[prompt_length] = []
        prompt_dict[prompt_length].append(input_tuple)

    batch_list = []
    for length_group, tuple_list in prompt_dict.items():
        n_batch = math.ceil(len(tuple_list)/test_batch_size)
        for b_i in range(n_batch):
            batch_data = tuple_list[b_i*test_batch_size : (b_i+1)*test_batch_size]
            batch_list.append(batch_data)

    for batch_data in tqdm(batch_list):
        x_list = [t[0] for t in batch_data]
        x = torch.cat(x_list, dim=0)
        with torch.no_grad():
            with ctx:
                y_out = model.generate(
                    x,
                    max_new_tokens,
                    temperature=1.0,
                    top_k=1
                )
                outcome_list = [decode(yy.tolist()) for yy in y_out]
                for i, outcome in enumerate(outcome_list):
                    _, prompt_len, idx_in_data = batch_data[i]

                    c_hat = outcome[prompt_len:].strip()
                    
                    gold_answer = str(answers[idx_in_data]).strip()
                    if algo_reason:
                        generated_answer = extract_answer_from_scratchpad(c_hat)
                    else:
                        generated_answer = extract_answer_from_output(c_hat, index_hint=index_hint)
                    if operator in ['multiply_nm', 'hex','+','tokenize']:
                        if zero_pad:
                            generated_answer = remove_zero_pad(generated_answer,reverse_c)
                        if reverse_c:
                            generated_answer = reverse_string(generated_answer)
                        if operator == 'tokenize':
                            gold_answer = gold_answer.replace(' ','')
                        if is_number(generated_answer):
                            if '.' in generated_answer:
                                generated_answer = float(generated_answer)
                            else:
                                generated_answer = int(generated_answer)
                            gold_answer=int(gold_answer)
                        else: # c_hat2 is not a number
                            generated_answer = str(generated_answer)
                    if generated_answer == gold_answer:
                        correct += 1
                        if verbose_correct:
                            print('outputs(o): ', outcome)
                            print(f'correct: {lines[idx_in_data]!r}{generated_answer!r}')
                    else:
                        if verbose:
                            print('outputs(x): ', outcome)
                            print(f'wrong: {generated_answer!r}')
                            print(f'correct: {gold_answer!r}')


    accuracy = correct / total * 100
    print(f"accuracy of {total} COPY examples: {correct}/{total} ({accuracy:.2f}%)")


    model.train()

    return accuracy

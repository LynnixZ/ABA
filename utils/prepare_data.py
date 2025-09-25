import os
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

def remove_zero_pad(s: str, reverse=True):
    # drop trailing (or leading if reverse=False) zeros/spaces, keep at least "0"
    chars = list(str(s))
    found = False
    if reverse:
        for i in range(len(chars) - 1, -1, -1):
            if not found:
                if chars[i].isdigit() and chars[i] == '0':
                    chars[i] = ''
                elif chars[i] != ' ':
                    found = True
        found = False
        for i in range(0, len(chars) - 1):
            if not found:
                if chars[i] == ' ':
                    chars[i] = ''
                else:
                    found = True
    else:
        for i in range(len(chars) - 1, -1, -1):
            if not found:
                if chars[i] == ' ':
                    chars[i] = ''
                else:
                    found = True
        found = False
        for i in range(0, len(chars) - 1):
            if not found:
                if chars[i] in (' ', '0'):
                    chars[i] = ''
                else:
                    found = True
    return ''.join(chars) if chars else '0'

def reverse_string(a: str) -> str:
    return str(a)[::-1]
def get_data_list(filename=None, operator='+', delim=None):
    import re
    data_list = []
    if not filename:
        return data_list  # return empty list if no filename is provided    

    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if '=' not in line:
            continue
        elif line.strip() == '=':
            continue
        # if first char is $, assume it's a delimiter
        if line[0] == '$':
            delim = '$'
        if delim:
            # remove delim from line
            line = line.replace(delim, '')
        # x1, x2 = line.strip().split(operator)
        if operator in ['+', '-', '*', '@','multiply_nm']:
            x1, x2 = re.split(r'[+\-x@*]', line.strip())
            x2, y = x2.split("=")
            data_list.append((int(x1), int(x2), int(y), operator))

        elif operator in ['parity','copy', 'reverse', 'hex']:
            #  "aaba=aaba"
            if '=' not in line:
                continue
            x, y = line.split('=')
            x = x.strip()
            y = y.strip()
            data_list.append((x, y, operator))

        elif operator == 'sort':
            if '=' not in line:
                continue
            left_part, right_part = line.split('=')
            left_part = left_part.strip()
            right_part = right_part.strip()
            
            pairs = left_part.split(',')
            x_list = []  
            a_list = []  
            
            for p in pairs:
                p = p.strip()
                if ':' not in p:
                    continue
                letter, val_str = p.split(':')
                letter = letter.strip()
                val_str = val_str.strip()
                
                a_list.append(letter)
                if val_str.isdigit():
                    x_list.append(int(val_str))
                else:
                    x_list.append(None)
            
            y_string = right_part  # "G,F" ...

            data_list.append((x_list, a_list, y_string, 'sort'))

        elif operator == 'multi_add':
            if '=' not in line:
                continue
            left_part, right_part = line.split('=')
            left_part = left_part.strip()   
            right_part = right_part.strip() 

            x_list = []
            for x in left_part.split('+'):
                x = x.strip()
                if x:  
                    x_list.append(int(x))
            
            y_string = right_part      
            operand_list = x_list
            data_list.append((operand_list, y_string, 'multi_add'))

    return data_list

def list_to_string(a):
    a = str(a)
    return a.replace(' ', '')


def truncate_to_n_digit(x, n=4):
    return math.floor(x * (10 ** n)) / (10 ** n)

def hint_index(num, start_index=0):
    # Convert the number to a list of its digits
    str_digits = list(str(num))
    letters = list(string.ascii_letters)  # ['a', 'b', 'c', ..., 'z', 'A', 'B', ..., 'Z']
    if start_index + len(str_digits) > len(letters):
        raise ValueError("The number is too large; it has more digits than available letters from the starting point.")
    result = [letters[start_index + i] + str(d) for i, d in enumerate(str_digits)]
    return ''.join(result)

def add_spaces(s):
    # add space if character is a digit or '=', else don't add space
    s = ''.join([c + ' ' if c.isdigit() or c in ['=','_', '$','.', '+', '-', '*', '('] else c for c in s])
    s = ' ' + s             
    if s[-1] == ' ':
        s = s[:-1]
    s = s.replace(' \n', '\n')
    return s

def get_pad_len(x0, x1, y=None):
    max_digits = max(len(list(str(x0))), len(list(str(x1))))
    y_digits = len(list(str(y))) if y is not None else 0
    extra_padding = 1  # we pad by 1 to allow space for a carry in the output
    return max(max_digits, y_digits) + extra_padding

def format_prompt(x0, x1, pad_len, y=None):
    def num2array(num: int, pad_len: int):
        str_digits = list(str(num).rjust(pad_len, '0'))
        digits = np.array([int(dig) for dig in str_digits])
        digits = np.array([[-100 - (i), d] for i, d in enumerate(digits)]).flatten()
        return digits

    x0_array = num2array(x0, pad_len)
    x1_array = num2array(x1, pad_len)
    if y is not None:
        y_array = num2array(y, pad_len)
        return  x0_array, x1_array,y_array
    else:
        return x0_array,  x1_array


def insert_spaces_at_edges(s, num_spaces_to_insert):
    if num_spaces_to_insert <= 0:
        return s
    # Randomly decide how many spaces to insert on the left and right
    num_left_spaces = random.randint(0, num_spaces_to_insert)
    num_right_spaces = num_spaces_to_insert - num_left_spaces
    return ' ' * num_left_spaces + s + ' ' * num_right_spaces


def insert_random_spaces(s, target_length):
    # Calculate how many spaces need to be inserted
    num_spaces_to_insert = target_length - len(s)
    if num_spaces_to_insert <= 0:
        return s
    positions = list(range(len(s) + 1))
    space_positions = random.choices(positions, k=num_spaces_to_insert)
    space_positions.sort()
    s_list = list(s)
    offset = 0
    for pos in space_positions:
        s_list.insert(pos + offset, ' ')
        offset += 1
    return ''.join(s_list)

def replace_trailing_zeros_with_spaces(s: str, reverse=True):
    chars = list(s)
    found_non_zero = False
    if reverse:
        for i in range(len(chars) - 1, -1, -1):
            c = chars[i]
            if not found_non_zero:
                if c == '0' or c ==' ':
                                       
                    chars[i] = ''
                else:
                                   
                    found_non_zero = True
                                        
            else:
                                     
                pass
                    
        found_non_zero = False

        for i in range(0, len(chars) - 1, 1):
            c = chars[i]
            if not found_non_zero:
                          
                if c ==' ':
                                        
                    chars[i] = ''
                else:
                                   
                    found_non_zero = True
                                        
            else:
                                     
                pass
    else:
        for i in range(len(chars) - 1, -1, -1):
            c = chars[i]
            if not found_non_zero:
                          

                if c ==' ':
                                        
                    chars[i] = ''
                else:
                                   
                    found_non_zero = True
                                        
            else:
                                     
                pass
                    
        found_non_zero = False

        for i in range(0, len(chars) - 1, 1):
            c = chars[i]
            if not found_non_zero:
                          
                if c ==' ' or c == '0':
                                        
                    chars[i] = ''
                else:
                                   
                    found_non_zero = True
                                        
            else:
                                     
                pass

    return ''.join(chars)

def insert_spaces_at_positions(s, digits_positions, total_length):
    result_list = [' '] * total_length
    for char, pos in zip(s, digits_positions):
        result_list[pos] = char
    return "".join(result_list)

def generate_fixed_space_positions(length, total_length):
    digits_positions = sorted(random.sample(range(total_length), k=length))
    return digits_positions

def insert_fixedspaces_at_numberlist(numlist,total_length):
    # Generate fixed space positions
    space_positions = generate_fixed_space_positions(len(numlist[0]), total_length)
    spaced_numlist = []
    for num in numlist:
        num_spaced = insert_spaces_at_positions(num, space_positions, total_length)
        spaced_numlist.append(num_spaced)
    return spaced_numlist

def remove_spaces(s):
    return s.replace(' ', '')

def insert_spaces(num_list, num_test_list, blank_space_in_equation_number, pad_before=False, 
                    fix_blank_space_position=True, blank_space_number_exact=True, zero_pad=True, reverse=True):
    # (Sabbaghi et al. 202e
    if pad_before:
        number_spaced, number_test_spaced=[], []
        if blank_space_number_exact:
            for a in num_list :
                num_spaces_to_insert = blank_space_in_equation_number - len(a)
                if num_spaces_to_insert < 0:
                    num_spaces_to_insert = 0  
                a = a+' ' * num_spaces_to_insert
                number_spaced.append(a)
            for a in num_test_list:
                num_spaces_to_insert = blank_space_in_equation_number - len(a)
                if num_spaces_to_insert < 0:
                    num_spaces_to_insert = 0  
                a = a+' ' * num_spaces_to_insert
                number_test_spaced.append(a)                               
        else:
            max_length = max([len(x) for x in num_list])
            total_length = random.randint(max_length, blank_space_in_equation_number)
            for a in num_list:
                num_spaces_to_insert = total_length - len(a)
                if num_spaces_to_insert < 0:
                    num_spaces_to_insert = 0  
                a = a+' ' * num_spaces_to_insert
                number_spaced.append(a)
            number_test_spaced= num_test_list
        return number_spaced, number_test_spaced
    
    """
    Core ABA Logic:
    """
    if fix_blank_space_position:
        # Ensure x1, x2, y are padded to same length
        max_number = max([len(str(x)) for x in num_list])
        num_list = [x.zfill(max_number) for x in num_list]
        if blank_space_number_exact: #ABA-fixed
            total_length = blank_space_in_equation_number
        else: #ABA-var
            total_length = random.randint(max_number, blank_space_in_equation_number)
        num_spaces_to_insert = total_length - max_number
        if num_spaces_to_insert > 0:
            spaced_numbers = insert_fixedspaces_at_numberlist(num_list, total_length)                       
        else:
            spaced_numbers = num_list
        # For test set, just pad with spaces at the end
        if blank_space_number_exact:
            number_test_spaced =[x.zfill(max_number)+' '* num_spaces_to_insert  for x in num_test_list]
        else:
            number_test_spaced = num_test_list
        if not zero_pad:
            # ABA required all numbers to be reversed or not reversed
            spaced_numbers = [replace_trailing_zeros_with_spaces(x, reverse=reverse) for x in spaced_numbers]
    # randomly insert spaces (Shen et al. 2023)    
    else:                     
        if blank_space_number_exact:                        
            spaced_numbers = [insert_random_spaces(x, blank_space_in_equation_number) for x in num_list]
            number_test_spaced = [insert_random_spaces(x, blank_space_in_equation_number) for x in num_test_list]                   
        else:
            spaced_numbers = []
            for i, x in enumerate(num_list):
                max_spaces = blank_space_in_equation_number - len(x)
                num_spaces = random.randint(0, max(0, max_spaces))
                x_spaced = insert_random_spaces(x, len(x) + num_spaces)
                spaced_numbers.append(x_spaced)
            number_test_spaced = num_test_list        
    return spaced_numbers, number_test_spaced
    
def generate_data_str(
    data_list, operator='+', format='plain', train=True, shuffle=False,
    fewshot=False, prompt=None, add_space=False, reverse_b=True, reverse_c=False, scratchpad_simple=False, index_hint=False,
    zero_pad=True, zeropad_max_length=None, 
    blank_space_in_equation_number=11,  fix_blank_space_position=False,
    blank_space_number_exact=True, 
    max_operand_number=10, operand_number_exact=True, pad_before=False, hard_mode=''
):
    if shuffle:
        random.shuffle(data_list)

    data_str = ''
    y_list = []
    prompt_list = []   
        
    if format == 'algo_reasoning' and operator == '*':
        for idx, data_tuple in enumerate(data_list):
            x1, x2, y = data_tuple[0], data_tuple[1], data_tuple[2]
            output_str = generate_scratchpad(x1, x2, blank_space_in_equation_number, max_operand_number, reverse=True, train=train, blank_space_exact=blank_space_number_exact,simple=scratchpad_simple)
            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str
            y_list.append(y)
            prompt_list.append(output_str)
        return data_str,prompt_list, y_list
    if format == 'algo_reasoning' and operator == 'parity':
        for idx, data_tuple in enumerate(data_list):
            x,y = data_tuple[0], data_tuple[1]
            output_str = generate_scratchpad_parity(x, y, blank_space_in_equation_number, blank_space_number_exact, fix_blank_space_position, train=train)
            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str
            prompt_list.append(output_str)
            y_list.append(y)
        return data_str,prompt_list, y_list
    
    for idx, data_tuple in enumerate(data_list):
        operator_in_tuple = data_tuple[-1]
        if operator_in_tuple in ['+', '-', '*','@']:   
            x1, x2, y_true = data_tuple[0], data_tuple[1], data_tuple[2]
            # Convert x1, x2, y to strings
            x1 = str(x1)
            x2 = str(x2)
            y = str(y_true)
            # Zero-pad numbers if zero_pad is True and max_number_length is provided
            if zero_pad and zeropad_max_length != None and zeropad_max_length is not 0:
                x1 = x1.zfill(zeropad_max_length)
                x2 = x2.zfill(zeropad_max_length)
                y = y.zfill(zeropad_max_length)

            elif zero_pad or (fix_blank_space_position and blank_space_in_equation_number is not None and blank_space_in_equation_number > 0):
                if operator=='+':
                    max_number = max(len(x1), len(x2)) + 1
                if operator=='*':
                    max_number = len(x1) + len(x2)
                x1 = x1.zfill(max_number)
                x2 = x2.zfill(max_number)
                y = y.zfill(max_number)
            # Reverse y if reverse_c is True or format indicates reversal
            if reverse_c:
                x1 = x1[::-1]
                x2 = x2[::-1]
                y = y[::-1]
            if index_hint:
                # Determine the maximum length of x1, x2, y
                max_length = max(len(x1), len(x2), len(y))
                # Randomly select a starting index within the valid range
                start_index = random.randint(0, len(string.ascii_letters) - max_length)
                # Apply hint_index with the same starting index for x1, x2, y
                x1 = hint_index(x1, start_index)
                x2 = hint_index(x2, start_index)
                y = hint_index(y, start_index)
            x1_test, x2_test = x1, x2
            if blank_space_in_equation_number is not None and blank_space_in_equation_number > 0:
                spaced_list, test_spaced_list = insert_spaces([x1, x2, y], [x1, x2], blank_space_in_equation_number,
                pad_before, fix_blank_space_position, blank_space_number_exact, zero_pad=zero_pad)
                x1, x2, y = spaced_list
                x1_test, x2_test = test_spaced_list

            if hard_mode == 'add_random_spaces':
                num_insertions = random.randint(1, max(1, len(x2) // 2))  # Insert up to 50% of the length of x2
                x2 = insert_random_spaces(x2, len(x2) + num_insertions)
            elif hard_mode == 'add_random_letter':
                letters = string.ascii_letters
                num_insertions = random.randint(1, max(1, len(x2) // 2))  # Insert up to 50% of the length of x2
                positions = random.sample(range(len(x2) + 1), num_insertions)
                for pos in sorted(positions, reverse=True):
                    random_letter = random.choice(letters)
                    x2 = x2[:pos] + random_letter + x2[pos:]
            elif hard_mode == 'position5_blank':
                if len(x2) >= 5:
                    x2 = x2[:5] + ' ' + x2[5:]
                    x1 = x1[:5] + ' ' + x1[5:]
                    y = y[:5] + ' ' + y[5:]
                if blank_space_number_exact:
                    x2_test = x2_test + ' ' 
                    x1_test = x1_test + ' ' 
            
            if reverse_b:
                x2 = x2[::-1]
                x2_test = x2_test[::-1]

            if train:
                output_str = f"${x1}{operator}{x2}={y}$\n"
            else: 
                output_str = f"${x1_test}{operator}{x2_test}="
                if fewshot:
                    output_str = prompt + output_str + '\n'
            if add_space:
                output_str=output_str.replace(' ', '_')
                output_str = add_spaces(output_str)
            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str
            prompt_list.append(output_str)
            y_list.append(y_true)

        elif operator_in_tuple == 'copy' or operator_in_tuple == 'multiply_nm' or operator_in_tuple == 'reverse' or operator_in_tuple == 'hex':
            if operator_in_tuple == 'copy' or operator_in_tuple == 'reverse':
                x, y_true = data_tuple[0], data_tuple[1]
                zero_pad = True # prevent remove 0 in insert_spaces function
                y = y_true
                if operator_in_tuple == 'reverse':
                    y = y[::-1]#will reverse it back after inserting spaces
            else:
                if operator_in_tuple == 'hex':
                    x, y_true = data_tuple[0], data_tuple[1]
                    max_number =len(str(x))
                else: # multiply_nm
                    x, x2, y_true = data_tuple[0], data_tuple[1], data_tuple[2]
                    max_number = max(len(x), len(x2)) + 2
                y=y_true
                if zero_pad and zeropad_max_length is not None and zeropad_max_length != 0:
                    x = x.zfill(zeropad_max_length)
                    y = y.zfill(zeropad_max_length)
                else:
                    x = x.zfill(max_number)
                    y = y.zfill(max_number)

            if operator_in_tuple == 'multiply_nm' or operator_in_tuple == 'hex':
                if reverse_c :
                    x = x[::-1]
                    y = y[::-1]
                    if operator_in_tuple == 'multiply_nm':
                        x2 = x2[::-1]
            x_test = x
            # add blank spaces
            if blank_space_in_equation_number is not None and blank_space_in_equation_number > 0:
                spaced_list, test_spaced_list = insert_spaces([x, y], [x], blank_space_in_equation_number,
                pad_before, fix_blank_space_position, blank_space_number_exact, zero_pad)
                x,y = spaced_list
                x_test = test_spaced_list[0]

            if operator_in_tuple == 'reverse':
                y = y[::-1]

            if operator_in_tuple == 'copy' or  operator_in_tuple == 'reverse' or  operator_in_tuple == 'hex':
                if train:
                    output_str = f"${x}={y}$\n"  
                else:    
                    output_str = f"${x_test}="
            elif operator_in_tuple == 'multiply_nm':
                if train:
                    output_str = f"${x}*{x2}={y}$\n"  
                else:    
                    output_str = f"${x_test}*{x2}="
            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str  
            prompt_list.append(output_str)
            y_list.append(y_true)

        elif operator_in_tuple == 'parity':
            x, y = data_tuple[0], data_tuple[1]
            if blank_space_in_equation_number is not None and blank_space_in_equation_number>0:
                if train:
                    max_spaces_x = blank_space_in_equation_number - len(x)
                    num_spaces_x = random.randint(0, max(0, max_spaces_x))
                    x = insert_random_spaces(x, len(x) + num_spaces_x)
                else:
                    x = x + ' ' * (blank_space_in_equation_number - len(x))
            else: 
                x = x
            if train:
                output_str = f"${x}={y}$\n"  
            else:    
                output_str = f"${x}="  
            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str
            prompt_list.append(output_str)
            y_list.append(y)

        elif operator_in_tuple == 'sort':
            x_list, a_list, y_str = data_tuple[0], data_tuple[1], data_tuple[2]
            x_list_numbers = []          
            a_list_numbers = []          
            a_list_placeholders = []   
            for x, a in zip(x_list, a_list):
                if x is not None:
                    x_list_numbers.append(x)
                    a_list_numbers.append(a)
                else:
                    a_list_placeholders.append(a)
            x_list_str = [str(x) for x in x_list_numbers]

            if zero_pad:
                if zeropad_max_length is not None and zeropad_max_length != 0:
                    x_list_str = [xx.zfill(zeropad_max_length) for xx in x_list_str]
                else:
                    max_len = max(len(xx) for xx in x_list_str)
                    x_list_str = [xx.zfill(max_len) for xx in x_list_str]
            elif fix_blank_space_position and blank_space_in_equation_number is not None and blank_space_in_equation_number > 0:
                max_len = max(len(xx) for xx in x_list_str)
                x_list_str = [xx.zfill(max_len) for xx in x_list_str]

            if reverse_c:
                x_list_str = [xx[::-1] for xx in x_list_str]

            x_list_test = []
            for xx in x_list_str:
                if fix_blank_space_position and not zero_pad:
                    xx_test = remove_zero_pad(xx,reverse_c)
                else:
                    xx_test = xx
                x_list_test.append(xx_test)

            if blank_space_in_equation_number is not None and blank_space_in_equation_number > 0:
                x_list_str, x_list_test = insert_spaces(x_list_str, x_list_test, blank_space_in_equation_number,
                pad_before, fix_blank_space_position, blank_space_number_exact, zero_pad)

            num_char_to_spaced_str = {}
            if len(a_list_numbers) == len(x_list_str):
                num_char_to_spaced_str = dict(zip(a_list_numbers, x_list_str))
            else:
                print("[Warning] x_list_spaced not found or mismatch. Check previous steps.")
                num_char_to_spaced_str = {a: str(x) for a, x in zip(a_list_numbers, x_list_numbers)}                         

            operands_processed = []
            if a_list_placeholders:
                for a in a_list_placeholders:
                    placeholders=" "*len(x_list_str[0])
            for a in a_list:
                if a in a_list_numbers:
                    formatted_str = num_char_to_spaced_str[a]
                    operands_processed.append(f"{a}:{formatted_str}")
                else:
                    operands_processed.append(f"{a}:{placeholders}")
                
            train_left_side = ",".join(operands_processed)
            operands_processed = []
            for letter, val_spaced in zip(a_list, x_list_test):
                operands_processed.append(f"{letter}:{val_spaced}")
            test_left_side = ",".join(operands_processed)

            if train:
                output_str = f"${train_left_side}={y_str}$\n"
            else:                         
                output_str = f"${test_left_side}="

            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str
            prompt_list.append(output_str) 
            y_list.append(y_str)

        elif operator_in_tuple == 'multi_add':

            operands_number,y_true = data_tuple[0], data_tuple[1]
            y_str = str(y_true)
            operands_processed = []
            test_left_side = []  
            operands=[str(x) for x in operands_number]

            if zero_pad:
                if zeropad_max_length is not None and zeropad_max_length != 0:
                    operands = [xx.zfill(zeropad_max_length) for xx in operands]
                else:
                    max_len = max(len(xx) for xx in operands)+1
                    operands = [xx.zfill(max_len) for xx in operands]
                    y_str = y_str.zfill(max_len)
            elif fix_blank_space_position and blank_space_in_equation_number is not None and blank_space_in_equation_number > 0:
                max_len = max(len(xx) for xx in operands)
                operands = [xx.zfill(max_len) for xx in operands]
                y_str = y_str.zfill(max_len)

            if reverse_c:
                operands = [xx[::-1] for xx in operands]
                y_str = y_str[::-1]
            operands.append(y_str)

            operands_test = []
            for xx in operands:
                if fix_blank_space_position and not zero_pad:
                    xx_test = remove_zero_pad(xx,reverse_c)
                else:
                    xx_test = xx
                operands_test.append(xx_test)
            
            if blank_space_in_equation_number is not None and blank_space_in_equation_number > 0:
                operands_processed, test_left_side = insert_spaces(operands, operands_test, blank_space_in_equation_number,
                pad_before, fix_blank_space_position, blank_space_number_exact, zero_pad)
                y_str = operands_processed[-1]
                operands_processed = operands_processed[:-1]

            # add spaces as blank operands
            current_len = len(operands_processed)                    
            if current_len < max_operand_number:
                if operand_number_exact:
                    spaces_to_add = max_operand_number - current_len
                else:
                    spaces_to_add = random.randint(0, max_operand_number - current_len)
                blank_space = ' ' * len(operands_processed[0]) 
                for _ in range(spaces_to_add):
                    operands_processed.append(blank_space)  
                    if operand_number_exact: test_left_side.append(blank_space)

            random.shuffle(operands_processed)
            train_left_side = "+".join(operands_processed)
            test_left_side = "+".join(test_left_side)

            if train:
                output_str = f"${train_left_side}={y_str}$\n"
            else:
                output_str = f"${test_left_side}="

            if idx == 0:
                data_str = output_str
            else:
                data_str += output_str
            prompt_list.append(output_str) 
            y_list.append(y_true)
    
    return data_str, prompt_list, y_list
###NOT FIXED YET
def generate_scratchpad(operand1, operand2, spaces1, spaces2,reverse=True,train=True,blank_space_exact=True,simple=True):
    op1_str = str(operand1)
    op2_str = str(operand2)
                             
    zero_pad_length = len(op1_str) + len(op2_str)
    op1_str = op1_str.zfill(zero_pad_length)
    blankspaces1 = spaces1 - zero_pad_length

                        
    if not train and not blank_space_exact:
        blankspaces1 = 0
    elif not blank_space_exact:
        blankspaces1 = random.randint(0, blankspaces1)

    if train:
        pos1 = generate_fixed_space_positions(len(op1_str), blankspaces1)
        spaced_op1 = insert_spaces_at_positions(op1_str, pos1)
    else:
        spaced_op1= " "*blankspaces1+op1_str
    
                         

    if not train:
        blankspaces2 = 0
    else:
        blankspaces2 = random.randint(0, spaces2 - len(op2_str))
    pos2 = generate_fixed_space_positions(len(op2_str), blankspaces2)
    spaced_op2 = insert_spaces_at_positions(op2_str, pos2)

           
    if reverse:
        spaced_op1 = spaced_op1[::-1]
        spaced_op2 = spaced_op2[::-1]
    header = f"${spaced_op2}*{spaced_op1}:\n"
    if not train:
        return header
    
                                            
    def format_with_spaces(val):
        s = str(val).zfill(zero_pad_length)
        s_spaced = insert_spaces_at_positions(s, pos1)
        if reverse:
            s_spaced = s_spaced[::-1]
        return s_spaced


    if simple:
        multiplication_lines = []
        intermediate_products = []
        digit_index = -1
                                      
        for ch in spaced_op2:
            if ch.strip():
                digit_index += 1
                current_digit = int(ch)
                prod = int(op1_str) * current_digit
                                                
                shifted_value = prod * (10 ** digit_index)
                shifted_formatted = format_with_spaces(shifted_value)
                line = f"{ch}-{shifted_formatted}"
                multiplication_lines.append(line)
                intermediate_products.append(shifted_formatted)
            else:
                blank_field = ' ' * len(spaced_op1)
                line = f" -{blank_field}"
                multiplication_lines.append(line)
                intermediate_products.append(blank_field)
        
                                            
        final_product = int(op1_str) * int(op2_str)
        final_formatted = format_with_spaces(final_product)
        addition_line = "+".join(intermediate_products) + "=" + final_formatted
        
                             
                                              
        scratchpad = header + "\n".join(multiplication_lines) + "\n>" + addition_line+ "$" 
        return scratchpad
    else:
        multiplication_lines = []
        cumulative = 0            
        digit_index = -1                

                              
        for ch in spaced_op2:
            if ch.strip():
                digit_index += 1
                current_digit = int(ch)
                prod = int(op1_str) * current_digit
                prod_str = str(prod).zfill(zero_pad_length)
                spaced_prod = insert_spaces_at_positions(prod_str, pos1)
                if reverse:
                    spaced_prod = spaced_prod[::-1]
                shifted_value = prod * (10 ** digit_index)
                shifted_str = str(shifted_value).zfill(zero_pad_length)
                shifted_prod = insert_spaces_at_positions(shifted_str, pos1)
                if reverse:
                    shifted_prod = shifted_prod[::-1]
                prev_cum_formatted = format_with_spaces(cumulative)
                new_cum = cumulative + shifted_value
                new_cum_formatted = format_with_spaces(new_cum)
                line = f"{ch}*{spaced_op1}={spaced_prod}>{shifted_prod}+{prev_cum_formatted}={new_cum_formatted},"
                cumulative = new_cum
            else:
                blank_field = ' ' * len(spaced_op1)
                prev_cum_formatted = format_with_spaces(cumulative)
                                                
                line = f" *{spaced_op1}={blank_field}>{blank_field}+{prev_cum_formatted}={prev_cum_formatted},"
            multiplication_lines.append(line)
        scratchpad = header + "\n".join(multiplication_lines )
        if scratchpad[-1] == ",":
            scratchpad = scratchpad[:-1] + "$"
        return scratchpad

def generate_scratchpad_parity(x,y, blank_space_in_equation_number, blank_space_number_exact, fix_blank_space_position, train=True):
    scratchpad='+' if x[0] == '1' else '-'
    for i, ch in enumerate(x[1:]):
        digit = int(ch)
        if digit ==1:
            scratchpad += "+" if scratchpad[-1] == '-' else '-'
        else:
            scratchpad += scratchpad[-1]

    if blank_space_in_equation_number is not None and blank_space_in_equation_number > 0:
        spaced_list, test_spaced_list = insert_spaces([x, scratchpad], [x], blank_space_in_equation_number=blank_space_in_equation_number,
        fix_blank_space_position=fix_blank_space_position, blank_space_number_exact=blank_space_number_exact)
        print(spaced_list, test_spaced_list)
        x,scratchpad = spaced_list
        x_test = test_spaced_list[0]
    if train:
        output_str = f"${x}={scratchpad}={y}$\n"  
    else:    
        output_str = f"${x_test}="  


    return output_str
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


# create and save meta file for a given vocabulary
def create_meta_file(vocabulary, input_data_str=None, tokenizer='char'):
    operators_str = string.punctuation
    if vocabulary == 'custom_input_data' and input_data_str:
        print(f"Input file {input_data_str[:100]} specified. Reading data from file...")
        data = input_data_str
        print(f"length of dataset in characters: {len(data):,}")
        vocabulary = 'custom_input_data'
    elif vocabulary == 'numbers_only':
        print(f"Creating meta file for numbers only...")
        data = string.digits + operators_str + ' \n'
    elif vocabulary == 'all_ascii_chars':
        print(f"Creating meta file for all reasonable characters...")
        data = string.ascii_lowercase + string.ascii_uppercase + string.digits + operators_str + ' \n'
    else:
        raise ValueError(f"Vocabulary {vocabulary} not supported!")
    
    if tokenizer == 'char':
        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        print("all the unique characters:", ''.join(chars))
        print(f"vocab size: {vocab_size:,}")

        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        def data_encoder(s):
            data_ids = [stoi[c] for c in s] # encoder: take a string, output a list of integers
            # convert to np array for efficiency
            data_ids = np.array(data_ids, dtype=np.uint16)
            return data_ids
        def data_decoder(l):
            return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        # data_ids = data_encoder(data)
        # print(f"data has {len(data_ids):,} tokens")
        # # convert to np array for efficiency
        # data_ids = np.array(data_ids, dtype=np.uint16)

        # save the meta information as well, to help us encode/decode later
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        meta_path = f'meta_{vocabulary}.pkl'

    elif tokenizer == 'gpt2':
        print("Ignore all above messages about the meta file!!!")
        print(f"Tokenizer specified as {tokenizer}. Loading it from tiktoken")
        enc = tiktoken.get_encoding("gpt2")
        # karpathy uses enc.encode_ordinary(), but since there is no decode_ordinary(), I'm switching to .encode()
        def data_encoder(s):
            data_ids = enc.encode(s, allowed_special={"<|endoftext|>"}) # encoder: take a string, output a list of integers
            # convert to np array for efficiency
            data_ids = np.array(data_ids, dtype=np.uint16)
            return data_ids

        def data_decoder(l):
            return enc.decode(l) # decoder: take a list of integers, output a string


        meta = {
            'vocab_size': enc.n_vocab,
        }
        meta_path = f'meta_pretrained_gpt2_tokenizer.pkl'

    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    return meta, meta_path, data_encoder, data_decoder

def get_encode_decode(meta_path=None, tokenizer='char'):
    import pickle, tiktoken
    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if meta_path and tokenizer == 'char':
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    elif tokenizer:
        print(f"Trying to load tiktoken's openAI {tokenizer} tokenizer")
        enc = tiktoken.get_encoding(f"{tokenizer}")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    return encode, decode


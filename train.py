"""
Train runner
"""

import os
import time
import math
import pickle
import pandas as pd
import yaml
import wandb

from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model.model import GPTConfig, GPT
from utils import *

print("PyTorch version:", torch.__version__)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print("TF32 support explicitly disabled for matmul and cuDNN.")

# I/O
out_dir = 'out'
resume_dir = ''
ckpt_path_name = 'ckpt.pt'
resume_iter = False
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb
wandb_entity = 'qzou24-university-of-wisconsin-madison'
wandb_log = False
wandb_project = 'Arithmetic'
wandb_run_name = f'run{time.time()}'
exp_name = 'default_exp_name'

# data
dataset = 'bal'
gradient_accumulation_steps = 1
test_batch_size = 128
batch_size = 12
block_size = 1024
train_data_path = 'train.bin'
val_data_path = 'val.bin'
train_both = False
data_ratio = 0.2
train_data_path2 = 'train_addition.bin'
val_data_path2 = 'val_addition.bin'

# evaluation toggles (only for quick loss/perf tracking during training)
eval_text = False
eval_text_data_path = None
eval_addition = False
start = " "
eval_addition_ar = False
start_ar = None
eval_other = False
start_other = None
other_operator = '+'
eval_addition_train = False
start_train = " "
# augmentation flags
reverse_b = False
reverse_c = False
algo_reason = False

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False
save_final = True
use_flash = True

# optimizer
learning_rate = 5e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = None

# system / dtype
backend = 'nccl'
device = 'cpu'
dtype = 'float16' if device == 'cpu' else ('bfloat16' if torch.cuda.is_bf16_supported() else 'float32')
compile_model = True
data_type = 'binary'
operator = '+'
data_shuffle = True
data_format = 'plain'
vocabulary = 'all_ascii_chars'
meta_path_specified = True
eps = 0
tokenizer = 'char'
zero_pad = True

zeropad_max_length = 0
blank_space_in_equation_number = 0
pad_answer = False
fix_blank_space_position = True
blank_space_exact = True
scratchpad_simple = False
index_hint = False
max_operand_number = 10
operand_number_exact_multiadd = False
pad_before = False

model_type = 'Transformer'
ff_proj = 'mlp'
normalization_layer = 'layernorm'
layer_norm_position = 'pre'
layer_norm_epsilon = 1e-6
drop_path_rate = 0.02
seed = 1337
time_limit = 30000

# length eval related configs (used only to set defaults that go into config.yaml, the real eval is in eval_final.py)
digit_test_number = 20
evaluation = True  # keep periodic loss eval
evaluation_heatmap = False
evaluation_length = True
evaluate_final = False
arithmetic_batch =True
# positional embedding and norms
positional_embedding = "learned"
no_pad_in_target = False
num_digit =20
test_limit=1000
hard_mode = ''
# -----------------------------------------------------------------------------
# pick up CLI overrides
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read())
if blank_space_in_equation_number == 0:
    blank_space_in_equation_number = None
if min_lr is None:
    min_lr = learning_rate/10

# -----------------------------------------------------------------------------
# set start file defaults for quick run (unchanged from your logic)
# -----------------------------------------------------------------------------
if digit_test_number == 0 and blank_space_in_equation_number:
    digit_test_number = blank_space_in_equation_number - 1
if start == "":
    if digit_test_number < 120:
        start = f"FILE:data/val/addition/test_exclude10/samelength/medium/add_samedigit_min10_max{digit_test_number}_limit1000_test.txt"
    else:
        start = f"FILE:data/val/addition/test_exclude10/samelength/medium/add_samedigit_min10_max{digit_test_number}_limit2000_test.txt"

# -----------------------------------------------------------------------------
# imports gated by model_type
# -----------------------------------------------------------------------------
assert model_type == 'Transformer', "This runner expects model_original.GPT; adjust if you use Monet."
# -----------------------------------------------------------------------------

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
set_seed(seed + seed_offset)

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
meta_path_specified = False
data_dir = os.path.join('data', dataset)
train_data_path = os.path.join(data_dir, train_data_path)
val_data = start[5:]

train_data_list = get_data_list(train_data_path, operator=operator)
val_data_list = get_data_list(val_data, operator=operator)
if eval_addition_train and start_train == " ":
    # specify the start_train to be oour train data file
    print("change start_train")
    start_train = f"FILE:{train_data_path}"

if algo_reason or data_format == 'algo_reasoning':
    arithmetic_batch = True
    algo_reason = True
    data_format = 'algo_reasoning'
    no_pad_in_target = True

train_data_str, train_data_list, _ = generate_data_str(
    train_data_list,
    operator=operator, format=data_format, train=True, shuffle=data_shuffle,
     scratchpad_simple=scratchpad_simple, index_hint=index_hint,
    reverse_b=reverse_b, reverse_c=reverse_c,
    zero_pad=zero_pad, zeropad_max_length=zeropad_max_length,
    blank_space_in_equation_number=blank_space_in_equation_number,
    fix_blank_space_position=fix_blank_space_position,
    blank_space_number_exact=blank_space_exact,
    max_operand_number=max_operand_number,
    operand_number_exact=operand_number_exact_multiadd,
    pad_before=pad_before, hard_mode=hard_mode
)

val_data_str, val_data_list, _ = generate_data_str(
    val_data_list,
    operator=operator, format=data_format, train=True, shuffle=data_shuffle,
    scratchpad_simple=scratchpad_simple, index_hint=index_hint,
    reverse_b=reverse_b, reverse_c=reverse_c,
    zero_pad=zero_pad, zeropad_max_length=zeropad_max_length,
    blank_space_in_equation_number=blank_space_in_equation_number,
    fix_blank_space_position=fix_blank_space_position,
    blank_space_number_exact=blank_space_exact,
    max_operand_number=max_operand_number,
    operand_number_exact=operand_number_exact_multiadd,
    pad_before=pad_before, hard_mode=hard_mode
)
# build vocab on-the-fly
meta, meta_path, data_encoder, data_decoder = create_meta_file(
    vocabulary=vocabulary, input_data_str=train_data_str, tokenizer=tokenizer
)
meta_vocab_size = meta['vocab_size']
batcher = DataLoader(
    data_encoder=data_encoder,train_data_list=train_data_list,
    val_data_list=val_data_list,train_data_str=train_data_str,
    val_data_str=val_data_str,arithmetic_batch=arithmetic_batch,
    batch_size=batch_size,
    block_size=block_size, operator=operator,
    no_pad_in_target=no_pad_in_target, device=device,
    device_type=device_type, pad_char=".", ignore_index=-1,
)
# one warm batch (optional debug)
_ =batcher.get_batch('train')
batcher.preview("train", data_decoder , max_show=10, print_raw=False)
batcher.preview("val",   data_decoder, max_show=10, print_raw=False)
# -----------------------------------------------------------------------------
# Model init / resume
# -----------------------------------------------------------------------------
if meta_path_specified:
    data_dir = os.path.join('data', dataset)
    meta_in_ds = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_in_ds):
        with open(meta_in_ds, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_in_ds})")
    else:
        meta_in_ds = None

model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=(meta_vocab_size if meta_vocab_size is not None else None),
    dropout=dropout, use_flash=use_flash, positional_embedding=positional_embedding,
    ff_proj=ff_proj, normalization_layer=normalization_layer,
    layer_norm_position=layer_norm_position, layer_norm_epsilon=layer_norm_epsilon,
    drop_path_rate=drop_path_rate
)

if init_from == 'scratch':
    if model_args['vocab_size'] is None:
        print("defaulting to vocab_size of 50304")
        model_args['vocab_size'] = 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    print(model.parameters)

elif init_from == 'resume':
    if resume_dir:
        checkpoint = torch.load(resume_dir, map_location=device)
    else:
        ckpt_path = os.path.join(out_dir, ckpt_path_name)
        checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_args = checkpoint['model_args']
    for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size','ff_proj','normalization_layer','layer_norm_position','layer_norm_epsilon']:
        model_args[k] = ckpt_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num'] if resume_iter else 0
    max_iters += iter_num
    best_val_loss = checkpoint.get('best_val_loss', 1e9)
    best_perplexity = checkpoint.get('best_perplexity', 1e9)
    best_accuracy = checkpoint.get('best_accuracy', -1)
else:
    raise ValueError("init_from must be 'scratch' or 'resume'")

model.to(device)

# amp scaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
else:
    best_val_loss = 1e9
    best_perplexity = 1e9
    best_accuracy = -1
    iter_num = 0

# compile
if compile_model:
    print("compiling the model...")
    model = torch.compile(model)

# DDP wrap
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# estimate_loss helper
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batcher.get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# cosine lr with warmup
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# wandb init
if wandb_log and master_process:
    try:
        wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name, config={k:globals()[k] for k in config_keys})
    except Exception as e:
        print(f"W&B init failed: {e}")
        wandb_log = False

# result dir and config dump
result_dir = get_results_dir({k:globals()[k] for k in config_keys})
if master_process:
    with open(os.path.join(result_dir, "config.yaml"), "w") as yaml_file:
        yaml.dump({k:globals()[k] for k in config_keys}, yaml_file, default_flow_style=False)

# encode/decode for optional prompting
encode, decode = get_encode_decode(meta_path, tokenizer=tokenizer)
if 'gpt2' in init_from:
    print_model_output(model, encode, decode, device=device)

# metrics buffers
result_dict = {
    'iter': [], 'train_loss': [], 'val_loss': [],
    'val_ppl': [], 'test_acc': [], 'train_acc': [],
    'time': [], 'iter_time_ms': [], 'peak_gpu_mem_gb': []
}
start_event = end_event = None
if torch.cuda.is_available():
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
total_time = 0.0
best_ood_acc = -1.0         # 对应 eval_addition 的 test_accuracy
    
# main loop
while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for g in optimizer.param_groups:
        g['lr'] = lr

    if evaluation and iter_num >= 0 and iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        ppl = None
        test_accuracy = None
        train_accuracy = None
        if eval_text:
            # optional if you set eval_text_data above
            with torch.no_grad():
                # you may have your own evaluate_text util; skip if not needed
                pass
        if eval_addition:
            cfg = {k:globals()[k] for k in config_keys}
            cfg['start'] = start
            test_accuracy = evaluate_addition_batch(
                cfg, model, ctx, encode, decode, verbose=True, num_digit=num_digit,
                zero_pad=zero_pad, reverse_b=reverse_b, reverse_c=reverse_c, algo_reason=algo_reason,
                 operator=operator, data_format=data_format,
                index_hint=index_hint, zeropad_max_length=zeropad_max_length,
                blank_space_in_equation_number=blank_space_in_equation_number,
                pad_answer=pad_answer, fix_blank_space_position=fix_blank_space_position,
                blank_space_exact=blank_space_exact, 
                operand_number_exact=operand_number_exact_multiadd, pad_before=pad_before, test_limit=test_limit, hard_mode=hard_mode
            )
        if eval_addition_train:
            cfg = {k:globals()[k] for k in config_keys}
            cfg['start'] = start_train if start_train.strip() else start
            train_accuracy= evaluate_addition_batch(
                cfg, model, ctx, encode, decode, verbose=True, num_digit=num_digit,
                zero_pad=zero_pad, reverse_b=reverse_b, reverse_c=reverse_c, algo_reason=algo_reason,
                 operator=operator, data_format=data_format,
                index_hint=index_hint, zeropad_max_length=zeropad_max_length,
                blank_space_in_equation_number=blank_space_in_equation_number,
                pad_answer=pad_answer, fix_blank_space_position=fix_blank_space_position,
                blank_space_exact=blank_space_exact, 
                operand_number_exact=operand_number_exact_multiadd, pad_before=pad_before,  test_limit=test_limit, hard_mode=hard_mode
            )

        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num, "train/loss": losses['train'], "val/loss": losses['val'],
                "lr": lr, "mfu": running_mfu*100,
                "ood_accuracy": test_accuracy if eval_addition else None,
                "iid_accuracy": train_accuracy if eval_addition_train else None
            }, step=iter_num)

        result_dict['iter'].append(iter_num)
        result_dict['train_loss'].append(losses['train'].item())
        result_dict['val_loss'].append(losses['val'].item())
        result_dict['val_ppl'].append(ppl.item() if ppl is not None else None)
        result_dict['test_acc'].append(test_accuracy if eval_addition else None)
        result_dict['train_acc'].append(train_accuracy if eval_addition_train else None)
        result_dict['time'].append(total_time)
        result_dict['iter_time_ms'].append(None)
        result_dict['peak_gpu_mem_gb'].append(None)

        # persist CSV only (no plots)
        pd.DataFrame(result_dict).to_csv(os.path.join(result_dir, 'result.csv'), index=False)

        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'best_perplexity': best_perplexity,
            'best_accuracy': best_accuracy,
            'config': {k:globals()[k] for k in config_keys},
        }
        if eval_addition and (test_accuracy is not None) and (test_accuracy > best_ood_acc):
            best_ood_acc = test_accuracy
            checkpoint['best_ood_acc'] = best_ood_acc
            torch.save(checkpoint, os.path.join(out_dir, ckpt_path_name.replace('.pt', '_best_ood.pt')))
            print(f"[best] OOD acc improved -> save {os.path.join(out_dir, ckpt_path_name.replace('.pt', '_best_ood.pt'))}")

        torch.cuda.empty_cache()
        if iter_num == 0 and eval_only:
            break

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_event.record()

    model.train()
    for micro_step in range(gradient_accumulation_steps):
        X, Y = batcher.get_batch('train')
        with ctx:
            logits, loss = model(X, Y)
        if torch.isnan(loss) or torch.isinf(loss):
            print("Loss is NaN or inf. Stopping training.")
            break
        # backward
        scaler.scale(loss).backward()
        torch.cuda.empty_cache()

    # clip
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    total_time += dt

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item()
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters or total_time > time_limit:
        break

if save_final and master_process:
    print(f"saving final checkpoint to {out_dir}")
    torch.save({
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'best_perplexity': best_perplexity,
        'best_accuracy': best_accuracy,
        'config': {k:globals()[k] for k in config_keys},
    }, os.path.join(out_dir, ckpt_path_name.split('.pt')[0]+'_final.pt'))

if wandb_log and master_process:
    wandb.finish()
if ddp:
    destroy_process_group()

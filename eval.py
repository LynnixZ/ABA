"""
Final evaluations after training.

Modes:
  - length   : accuracy vs. digit length
  - grid     : 2D grid for multi_add or sort (rows = operand/array size, cols = digit length)
  - heatmap  : (+) digit1 x digit2 heatmap
  - final    : run a single final test file
"""

import os
import csv
import math
import argparse
import numpy as np
import pandas as pd
import yaml
import torch
from contextlib import nullcontext

from utils import *
from model.model import GPTConfig, GPT

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_cfg_and_model(out_dir, ckpt_name):
    cfg_path = None
    for name in os.listdir(out_dir):
        if name.endswith('_config.yaml'):
            cfg_path = os.path.join(out_dir, name)
    if cfg_path is None:
        cand = os.path.join(out_dir, 'config.yaml')
        cfg_path = cand if os.path.exists(cand) else None

    config = {}
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config: {cfg_path}")
    else:
        print("Warn: config.yaml not found; using args/defaults only.")

    device = config.get('device', 'cpu')
    dtype = config.get('dtype', 'float32')
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}.get(dtype, torch.float32)
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    ckpt_path = os.path.join(out_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        if os.path.exists(ckpt_name):
            ckpt_path = ckpt_name
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint.get('model', checkpoint)
    up = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(up):
            state_dict[k[len(up):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return config, model, ctx


def build_codec_from_testfile(vocabulary, tokenizer, test_file_path):
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_data_str = f.read()
    meta, meta_path, data_encoder, data_decoder = create_meta_file(
        vocabulary=vocabulary, input_data_str=test_data_str, tokenizer=tokenizer
    )
    encode, decode = get_encode_decode(meta_path, tokenizer=tokenizer)
    return encode, decode, data_encoder, data_decoder


def _pick(arg_val, cfg, key, default=None):
    return arg_val if arg_val is not None else cfg.get(key, default)


def resolve_eval_options(args, config):
    return {
        "operator": _pick(args.operator, config, 'operator', '+'),
        "digit_max": _pick(args.digit_test_number, config, 'digit_test_number', 20),
        "data_format": config.get('data_format', 'plain'),
        "tokenizer": config.get('tokenizer', 'char'),
        "vocabulary": config.get('vocabulary', 'all_ascii_chars'),
        "zero_pad": config.get('zero_pad', False),
        "reverse_b": config.get('reverse_b', False),
        "reverse_c": config.get('reverse_c', False),
        "algo_reason": config.get('algo_reason', False),
        "blank_space_in_equation_number": config.get('blank_space_in_equation_number', None),
        "pad_answer": config.get('pad_answer', False),
        "index_hint": _pick(getattr(args, 'index_hint', None), config, 'index_hint', False),
        "zeropad_max_length": _pick(getattr(args, 'zeropad_max_length', None), config, 'zeropad_max_length', None),
        "operand_number_exact_multiadd": _pick(getattr(args, 'operand_number_exact_multiadd', None), config, 'operand_number_exact_multiadd', True),
        "pad_before": _pick(getattr(args, 'pad_before', None), config, 'pad_before', False),
        "fix_blank_space_position": config.get('fix_blank_space_position', False),
        "blank_space_exact": config.get('blank_space_exact', False),
        "hard_mode": config.get('hard_mode', ''),
    }


def eval_length(args, config, model, ctx):
    opts = resolve_eval_options(args, config)
    operator = opts["operator"]
    digit_max = opts["digit_max"]
    data_format = opts["data_format"]
    tokenizer = opts["tokenizer"]
    vocabulary = opts["vocabulary"]
    result_dir = get_results_dir(config)

    acc = {}
    for d in range(1, digit_max + 1):
        if operator in ['+','*']:
            test_path = f"data/val/addition/finaltest/multi_digit_test_samelength/medium/add_samedigit_min{d}_max{d}_limit500_test.txt"
            if data_format == 'algo_reasoning':
                test_path = f"data/val/addition/finaltest/multi_digit_test_samelength/verysmall/add_diffdigit_{d}and{d}_limit100.txt"
        elif operator == "multiply_nm":
            test_path = f"data/newtask/eval/multiply/multiply_{d}.txt"
        elif operator in ["parity","binarysum","copy","reverse","oneDigitSort","hex"]:
            if operator == "hex":
                test_path = f"data/newtask/eval/hexadecimal/hexadecimal_{d}.txt"
            else:
                test_path = f"data/newtask/eval/{operator}/{operator}_{d}.txt"
        else:
            raise ValueError(f"Unsupported operator for length mode: {operator}")

        if not os.path.exists(test_path):
            print(f"[length] missing file, skip: {test_path}")
            acc[d] = math.nan
            continue

        encode, decode, _, _ = build_codec_from_testfile(vocabulary, tokenizer, test_path)
        cfg = dict(config); cfg['start'] = f"FILE:{test_path}"
        num_digit = d + 3  # leave some margin

        try:
            digit_accuracy = evaluate_addition_batch(
                cfg, model, ctx, encode, decode, verbose=True, num_digit=num_digit,
                zero_pad=opts["zero_pad"], reverse_b=opts["reverse_b"], reverse_c=opts["reverse_c"],
                algo_reason=opts["algo_reason"], operator=operator, data_format=data_format,
                index_hint=opts["index_hint"], zeropad_max_length=opts["zeropad_max_length"],
                blank_space_in_equation_number=opts["blank_space_in_equation_number"],
                pad_answer=opts["pad_answer"], fix_blank_space_position=opts["fix_blank_space_position"],
                blank_space_exact=opts["blank_space_exact"],
                operand_number_exact=opts["operand_number_exact_multiadd"], pad_before=opts["pad_before"], hard_mode=opts["hard_mode"]
            )
        except Exception as e:
            print(f"[length] error at digit {d}: {e}")
            digit_accuracy = math.nan

        acc[d] = digit_accuracy

    csv_path = os.path.join(result_dir, "digit_accuracy_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["digit", "accuracy"])
        for d in range(1, digit_max+1):
            w.writerow([d, acc[d]])
    print(f"[length] CSV saved: {csv_path}")

    if args.plot and plt is not None:
        xs = list(range(1, digit_max+1))
        ys = [acc[d] for d in xs]
        plt.figure(figsize=(6,4))
        valid = ~np.isnan(ys)
        if np.any(valid):
            plt.plot(np.array(xs)[valid], np.array(ys)[valid], marker='o')
        plt.xlabel("Digit"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Digit Length")
        plt.ylim(0, 100); plt.grid(True)
        fig_path = os.path.join(result_dir, "digit_accuracy_plot.png")
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"[length] plot saved: {fig_path}")


def eval_grid(args, config, model, ctx):
    opts = resolve_eval_options(args, config)
    operator = args.operator
    assert operator in ["multi_add","sort"], "grid mode supports operator 'multi_add' or 'sort'."
    digit_max = opts["digit_max"]
    data_format = opts["data_format"]
    tokenizer = opts["tokenizer"]
    vocabulary = opts["vocabulary"]
    result_dir = get_results_dir(config)

    accuracy_matrix = np.zeros((digit_max, digit_max)) * np.nan

    for d in range(1, digit_max+1):
        for oc in range(1, digit_max+1):
            if operator == "multi_add":
                test_file = f"data/newtask/eval/multi_add/multi_add_d{d}_o{oc}_test.txt"
                nd = (opts["blank_space_in_equation_number"] + 5) if (opts["blank_space_in_equation_number"] and opts["blank_space_exact"]) else (d + 5)
            else:
                test_file = f"data/newtask/eval/sort/test_sort_{d}digit_{oc}array.txt"
                nd = oc + 5

            if not os.path.exists(test_file):
                print(f"[grid] missing file: {test_file}")
                continue

            encode, decode, _, _ = build_codec_from_testfile(vocabulary, tokenizer, test_file)
            cfg = dict(config); cfg['start'] = f"FILE:{test_file}"
            try:
                acc = evaluate_addition_batch(
                    cfg, model, ctx, encode, decode, verbose=True, num_digit=nd,
                    zero_pad=opts["zero_pad"], reverse_b=opts["reverse_b"], reverse_c=opts["reverse_c"],
                    algo_reason=opts["algo_reason"], operator=operator, data_format=data_format,
                    index_hint=opts["index_hint"], zeropad_max_length=opts["zeropad_max_length"],
                    blank_space_in_equation_number=opts["blank_space_in_equation_number"],
                    pad_answer=opts["pad_answer"], fix_blank_space_position=opts["fix_blank_space_position"],
                    blank_space_exact=opts["blank_space_exact"],
                    operand_number_exact=opts["operand_number_exact_multiadd"], pad_before=opts["pad_before"],
                )
            except Exception as e:
                print(f"[grid] error d={d} oc={oc}: {e}")
                acc = math.nan

            accuracy_matrix[oc-1, d-1] = acc

    csv_path = os.path.join(result_dir, "digit_accuracy_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["Operand/ArraySize \\ Digit"] + [str(d) for d in range(1, digit_max+1)]
        w.writerow(header)
        for oc in range(1, digit_max+1):
            row = [str(oc)] + [accuracy_matrix[oc-1, d-1] for d in range(1, digit_max+1)]
            w.writerow(row)
    print(f"[grid] CSV saved: {csv_path}")

    if args.plot and plt is not None:
        plt.figure(figsize=(8,6))
        disp = np.nan_to_num(accuracy_matrix, nan=-1.0)
        im = plt.imshow(disp, origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=100)
        plt.colorbar(im, label="Accuracy (%)")
        plt.title(f"{operator} Accuracy Grid")
        plt.xlabel("Digit Length"); plt.ylabel("Operand Count / Array Size")
        out_path = os.path.join(result_dir, f"{operator}_grid_heatmap.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"[grid] heatmap saved: {out_path}")


def eval_heatmap(args, config, model, ctx):
    opts = resolve_eval_options(args, config)
    operator = opts["operator"]
    assert operator == '+', "heatmap mode supports '+' only."
    digit_max = opts["digit_max"]
    data_format = opts["data_format"]
    tokenizer = opts["tokenizer"]
    vocabulary = opts["vocabulary"]
    result_dir = get_results_dir(config)

    H = np.zeros((digit_max, digit_max)) * np.nan

    for d1 in range(1, digit_max+1):
        for d2 in range(1, digit_max+1):
            test_file = f"data/val/addition/finaltest/heatmap/add_diffdigit_{d1}and{d2}_limit100.txt"
            if not os.path.exists(test_file):
                print(f"[heatmap] missing file: {test_file}")
                continue
            encode, decode, _, _ = build_codec_from_testfile(vocabulary, tokenizer, test_file)
            cfg = dict(config); cfg['start'] = f"FILE:{test_file}"
            nd = max(d1, d2) + 3 if not opts["algo_reason"] else d2
            try:
                acc = evaluate_addition_batch(
                    cfg, model, ctx, encode, decode, verbose=True, num_digit=nd,
                    zero_pad=opts["zero_pad"], reverse_b=opts["reverse_b"], reverse_c=opts["reverse_c"],
                    algo_reason=opts["algo_reason"], operator=operator, data_format=data_format,
                    index_hint=opts["index_hint"], zeropad_max_length=opts["zeropad_max_length"],
                    blank_space_in_equation_number=opts["blank_space_in_equation_number"],
                    pad_answer=opts["pad_answer"], fix_blank_space_position=opts["fix_blank_space_position"],
                    blank_space_exact=opts["blank_space_exact"],
                    operand_number_exact=opts["operand_number_exact_multiadd"], pad_before=opts["pad_before"],
                )
            except Exception as e:
                print(f"[heatmap] error ({d1},{d2}): {e}")
                acc = math.nan
            H[d1-1, d2-1] = acc

    csv_path = os.path.join(result_dir, "heatmap_accuracy_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["operand1_digit","operand2_digit","accuracy"])
        for d1 in range(1, digit_max+1):
            for d2 in range(1, digit_max+1):
                w.writerow([d1, d2, H[d1-1, d2-1]])
    print(f"[heatmap] CSV saved: {csv_path}")

    if args.plot and plt is not None:
        plt.figure(figsize=(10,8))
        im = plt.imshow(np.nan_to_num(H, nan=-1.0), origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=100)
        plt.colorbar(im, label="Accuracy (%)")
        plt.xlabel("Operand2 Digit Length"); plt.ylabel("Operand1 Digit Length")
        plt.title("Accuracy Heatmap (+)")
        fig_path = os.path.join(result_dir, "digit_accuracy_heatmap.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"[heatmap] plot saved: {fig_path}")


def eval_final_single(args, config, model, ctx):
    if not args.final_file:
        raise ValueError("--final_file is required for mode=final")
    opts = resolve_eval_options(args, config)
    operator = opts["operator"]
    data_format = opts["data_format"]
    tokenizer = opts["tokenizer"]
    vocabulary = opts["vocabulary"]

    test_file = args.final_file
    if not os.path.exists(test_file):
        raise FileNotFoundError(test_file)

    encode, decode, _, _ = build_codec_from_testfile(vocabulary, tokenizer, test_file)
    cfg = dict(config); cfg['start'] = f"FILE:{test_file}"

    num_digit = args.num_digit or opts["digit_max"]
    acc = evaluate_addition_batch(
        cfg, model, ctx, encode, decode, verbose=True, num_digit=num_digit,
        zero_pad=opts["zero_pad"], reverse_b=opts["reverse_b"], reverse_c=opts["reverse_c"],
        algo_reason=opts["algo_reason"], operator=operator, data_format=data_format,
        index_hint=opts["index_hint"], zeropad_max_length=opts["zeropad_max_length"],
        blank_space_in_equation_number=opts["blank_space_in_equation_number"],
        pad_answer=opts["pad_answer"], fix_blank_space_position=opts["fix_blank_space_position"],
        blank_space_exact=opts["blank_space_exact"],
        operand_number_exact=opts["operand_number_exact_multiadd"], pad_before=opts["pad_before"],
    )

    result_dir = get_results_dir(config)
    csv_path = os.path.join(result_dir, "final_accuracy.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["final_file","num_digit","final_accuracy"])
        w.writerow([test_file, num_digit, acc])
    print(f"[final] CSV saved: {csv_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=str, default='out')
    p.add_argument('--ckpt', type=str, default='ckpt.pt')
    p.add_argument('--mode', type=str, choices=['length','grid','heatmap','final'], required=True)
    p.add_argument('--operator', type=str, default=None)
    p.add_argument('--digit_test_number', type=int, default=None)
    p.add_argument('--final_file', type=str, default=None)
    p.add_argument('--num_digit', type=int, default=None)
    p.add_argument('--plot', action='store_true')

    # new/overridable flags used by evaluate_addition_batch
    p.add_argument('--index_hint', action='store_true')
    p.add_argument('--zeropad_max_length', type=int, default=None)
    p.add_argument('--operand_number_exact_multiadd', action='store_true')
    p.add_argument('--pad_before', action='store_true')

    args = p.parse_args()

    config, model, ctx = load_cfg_and_model(args.out_dir, args.ckpt)
    if args.operator is not None:
        config['operator'] = args.operator

    set_seed(config.get('seed', 1337))

    if args.mode == 'length':
        eval_length(args, config, model, ctx)
    elif args.mode == 'grid':
        eval_grid(args, config, model, ctx)
    elif args.mode == 'heatmap':
        eval_heatmap(args, config, model, ctx)
    elif args.mode == 'final':
        eval_final_single(args, config, model, ctx)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()

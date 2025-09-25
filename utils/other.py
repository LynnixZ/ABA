import os
import torch
import numpy as np
import random

# function to set seed for all random number generators
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # to make sure GPU runs are deterministic even if they are slower set this to True
    torch.backends.cudnn.deterministic = False
    # warning: this causes the code to vary across runs
    torch.backends.cudnn.benchmark = True
    print("Seeded everything: {}".format(seed))
    
def get_results_dir(config):
    results_dir = os.path.join(
        config['out_dir'],
        config['exp_name'] if config['exp_name'] != 'default_exp_name' else config['wandb_run_name']
    )
    # Reuse existing dir; no numbered suffix.
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def print_model_output(model, encode, decode, max_new_tokens=50, temperature=0.8, top_k=200, device='cuda', device_type='cuda', ptdtype=torch.float16, start="Twinkle twinkle little star"):
    num_samples = 1
    # encode the beginning of the prompt
    print('\n-----------------------------------------------------------------------------------------------')
    print(f"Prompting model with {start} for {max_new_tokens} tokens")
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('-----------------------------------------------------------------------------------------------\n')
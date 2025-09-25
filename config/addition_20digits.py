# Model and Training Configuration
out_dir = f'out/'  # Dynamic output directory based on layers and mixings
eval_interval = 1000  # As specified
eval_iters = 10
log_interval = 100
digit_test_number=200
arithmetic_batch=True

always_save_checkpoint = False
wandb_log = False  # Disable wandb logging per command line
wandb_project = 'arithmetic'
wandb_run_name = 'addition_20'

data_type = 'text'
data_format = 'reverse'
operator = '+'
dataset = 'bal'
batch_size = 128  # Adjusted as per command
block_size = 1024
train_data_path = '+_maxLen_20_limit_1000000_train_minReq_0.txt'
start = ''
start_train="FILE:data/bal/+_maxLen_20_limit_1000_test_minReq_0.txt"                        
ckpt_path_name = f'addition_20.pt'  # Checkpoint name as per layers and mixings
eval_addition = True
eval_addition_train = True
num_digit=200
# Model architecture settings
n_layer = 6  # Layer count as per command line
n_embd = 384  # Updated embedding size
n_head = 6
dropout = 0.2
positional_embedding = 'learned'
bias=True
learning_rate = 0.0005  # Updated learning rate
gradient_accumulation_steps = 4 # Added as per command line

max_iters = 5000
lr_decay_iters = 5000
beta2 = 0.99

warmup_iters = 2000
device = 'cuda'  # As specified

# Training settings for reverse and padding
reverse_c = True
reverse_ab = True
index_hint = False
zero_pad_in_target = False
zero_pad_in_training = True
max_number_length = 0
blank_space_in_equation_number = 201  # Placeholder; ensure `blank_space` is defined
pad_before = False  # Placeholder; ensure `pad_answer` is defined
fix_blank_space_position = True
blank_space_exact=True
blank_space_split_number=True

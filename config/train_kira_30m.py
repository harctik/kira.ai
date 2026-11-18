# config/train_kira_30m.py
# nanoGPT configuration for Kira 30M (Trained on 100% Human Wikipedia/Books)

out_dir = 'out-kira-30m'
eval_interval = 250
eval_iters = 40
log_interval = 10

# Saving checkpoint
always_save_checkpoint = False

# Dataset configuration
dataset = 'kira'
gradient_accumulation_steps = 4
batch_size = 32
block_size = 512

# Model architecture (~31.2M parameters)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

# Custom vocabulary size (BPE trained from scratch, NOT GPT-2's 50,257)
vocab_size = 8192

# Optimizer & LR schedule
learning_rate = 5e-4
max_iters = 8000
lr_decay_iters = 8000
min_lr = 5e-5
beta2 = 0.95
warmup_iters = 300

# Device settings
device = 'cuda'
compile = True  # set to False if on PyTorch < 2.0 or Windows

# config/train_kira_15m.py
# nanoGPT configuration for Kira 15M (Trained on 100% Human Wikipedia/Books)

out_dir = 'out-kira-15m'
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

# Model architecture (~15.4M parameters)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

# Custom vocabulary size (BPE trained from scratch, NOT GPT-2's 50,257)
# nanoGPT will automatically override with meta['vocab_size'] if meta.pkl is present
vocab_size = 8192

# Optimizer & LR schedule
learning_rate = 6e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 6e-5
beta2 = 0.95
warmup_iters = 200

# Device settings
device = 'cuda'
compile = True  # set to False if on PyTorch < 2.0 or Windows

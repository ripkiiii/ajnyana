import os, math, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from dataclasses import dataclass

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(BASE, 'checkpoints', 'latest.pt')
TOK_DIR   = os.path.join(BASE, 'tokenizer')
VAL_TXT   = os.path.join(BASE, 'data', 'processed', 'val.txt')
VAL_BIN   = '/tmp/ajnyana_val_eval.bin'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device    : {device}')
print(f'Checkpoint: {CKPT_PATH}')

# Load tokenizer
tokenizer = ByteLevelBPETokenizer(
    os.path.join(TOK_DIR, 'vocab.json'),
    os.path.join(TOK_DIR, 'merges.txt'),
)
EOT_ID = tokenizer.token_to_id('<|endoftext|>')

# Tokenize val.txt
print('Tokenizing val.txt...')
ids = []
with open(VAL_TXT, 'r', encoding='utf-8') as f:
    for line in f:
        doc = line.strip()
        if not doc:
            continue
        ids.extend(tokenizer.encode(doc).ids)
        ids.append(EOT_ID)
arr = np.array(ids, dtype=np.uint16)
arr.tofile(VAL_BIN)
print(f'Val tokens: {len(arr):,}')

# Model
@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 16000
    n_layer:    int = 6
    n_head:     int = 4
    n_embd:     int = 256
    dropout:    float = 0.0

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        nh, hs = self.n_head, C // self.n_head
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(self.act(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=False)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=False)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=False),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x    = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# Load checkpoint
ckpt   = torch.load(CKPT_PATH, map_location=device, weights_only=False)
config = ckpt['config'] if isinstance(ckpt['config'], GPTConfig) else GPTConfig(**ckpt['config'])
model  = GPT(config)
model.load_state_dict(ckpt['model'])
model.to(device)
model.eval()
step = ckpt.get('step') or ckpt.get('iter') or ckpt.get('iteration', '?')
print(f'Loaded step {step} | val {ckpt["val_loss"]:.4f}')

# Eval
BLOCK_SIZE = config.block_size
EVAL_BATCH = 32
val_data   = np.frombuffer(open(VAL_BIN, 'rb').read(), dtype=np.uint16).astype(np.int64)
val_tensor = torch.from_numpy(val_data)

total_loss, total_steps = 0.0, 0
with torch.no_grad():
    for start in range(0, len(val_tensor) - BLOCK_SIZE - 1, BLOCK_SIZE * EVAL_BATCH):
        xs, ys = [], []
        for b in range(EVAL_BATCH):
            s = start + b * BLOCK_SIZE
            if s + BLOCK_SIZE + 1 > len(val_tensor):
                break
            xs.append(val_tensor[s : s + BLOCK_SIZE])
            ys.append(val_tensor[s + 1 : s + BLOCK_SIZE + 1])
        if not xs:
            break
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        _, loss = model(x, y)
        total_loss  += loss.item()
        total_steps += 1

avg_loss   = total_loss / total_steps
perplexity = math.exp(avg_loss)

print(f'\n=== Ajnyana-1 Eval Results ===')
print(f'Val windows : {total_steps:,}')
print(f'NLL loss    : {avg_loss:.4f}')
print(f'Perplexity  : {perplexity:.2f}')

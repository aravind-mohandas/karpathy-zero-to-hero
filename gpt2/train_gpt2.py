from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import inspect
import tiktoken
import numpy as np
# ----------------------------------------

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ['train', 'val'], f"invalid split: {split}"

        # get the shard filenames:
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"loading {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):

        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T* self.num_processes + 1) >= len(self.tokens):
            self.current_shard  = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank # reset for next epoch
        return x, y

class CausalSelfAttention(nn.Module):

    # a more efficient way of implementing compared to what we did for nanogpt

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # standard deviation grows inside the residual stream, so we need to multiply by a scaling factor
        # if there are hundred residual layers, the std will grow by sqrt(100) = 10 times
        self.c_proj.NANOGPT_SCALE_INIT = 1
        #regulatization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Defining the mask for tril
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.size()   # batch size, sequence length/ block size, embedding size(n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is number of heads, hs is head size and C number of channels (n_embd = nh*ns)
        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim = 2) # each is B, T, C
        
        # converting into multiple heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # calculating attentiong (Now all calculations will be calculated for the dimensions B and nh together)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, hs)
        # we will use flash attention instead
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # reassemble all head outputs side by side (B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') 
        # gelu activation function with tanh approximation is used in gpt2, althought now the exact version is more common
        # gelu is better than relu since it does not give hard zeroes, allowing small negative values to pass through
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence lenght
    vocab_size: int = 50257 # number of tokens: 50000 BPE + 256 bytes tokens + 1 </endoftext> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # We will design it similar to Hugging Face's schema
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing 
        self.transformer.wte.weight = self.lm_head.weight
        # this is following the impementation in both gpt2 and attention is all you need paper

        # initialize parameters
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2*self.config.n_layer)**-0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            # typically follows dividing by sqrt(fan_in) but gpt2 uses fixed 0.02
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward, model block size is {self.config.block_size}"
        # forward tht token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape T
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)

        # going through the transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None 

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # converting logits to shape (B*T, vocab_size) and targets to shape (B*T,) for cross entropy calculation   
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model from Hugging Face model hub."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print('Loading weights from pretrained gpt', model_type)

        # n_layer, n_head and n_embd are determined by the model type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('attn.bias')] # discard the buffer masks

        # initialize a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all the parameters are aligned and match the names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.masked_bias')] # discard the buffer masks
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # discard the buffer masks
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # openai checkpoints use conv1d , we need to transpose those weight matrices
        assert len(sd_keys_hf) == len(sd_keys), f'mismatch in number of parameters: {len(sd_keys_hf)} vs {len(sd_keys)}'
        for k in sd_keys_hf:
            if any(k.endswith(t) for t in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters 
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups: any parametres that have dimension >= 2 will be decayed, others will not be decayed
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print("num decayed parameter tensors: %d, num non-decayed parameter tensors: %d" % (len(decay_params), len(nodecay_params)))

        fused_available = 'fused ' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print("using fused AdamW: %s" % use_fused, "Fused available: %s" % fused_available)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas = (0.9, 0.95), eps = 1e-8)
        return optimizer
#------------

# enabling tf32, so the matrix multiplications are done in float32 for faster computation
# tf32 crops out the precision compared to float32
# fp16 has a reduced range compared to float32, but bf16 maintatins higher range but lower precision
  
# simple launch: python train_gpt2.py
# ddp launch: torchrun --standalone --nproc_per_node=2 train_gpt2.py


# Running the training loop

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# setting up ddp
# torchrun command sets environment variables RANK, LOCAL_RANK and WORLD_SIZE
# RANK is the rank of the process
# LOCAL_RANK is the rank of the process on the node
# WORLD_SIZE is the total number of processes
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "cuda is not available"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ.get('RANK', -1))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK', -1))
    ddp_world_size = int(os.environ.get('WORLD_SIZE', -1))
    print("WORLD_SIZE", ddp_world_size)
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    master_process = True
    print("using device", device)



torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B = 16 # micro batch size
T = 512 # sequence length
assert total_batch_size % (B*T*ddp_world_size) == 0, "total_batch_size must be divisible by (B*T*ddp_world_size)"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size) # because we are using ddp, we need to divide by the number of processes
if master_process: # there is no need to print this for every process
    print("total desired barch size", total_batch_size)
    print(f"calculated grad accumulation steps: {grad_accum_steps}")


train_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split = "train")
val_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split = "val")

torch.set_float32_matmul_precision('high')


# create model
model = GPT(GPTConfig(vocab_size = 50304)) # making it closer to a number that is more divisible by powers of 2
model.to(device)
use_compile = False # it is intergfering with generation
if use_compile:
    model = torch.compile(model)
# model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # uncomment for final training
max_steps = 19073
# warmup_steps = 10
# max_steps = 200
def get_lr(it):
    # 1. linear warmup for the first warmup_steps steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and decays to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas = (0.9, 0.95), eps = 1e-8) # following the optimzations in gpt3
optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = max_lr, device = device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model
enc = tiktoken.get_encoding("gpt2")
# currently using as a black box, need to optimize

# create a log_directory we will create checkpoints to and log to 
log_dir = "log"
os.makedirs(log_dir, exist_ok = True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    pass


import time
for step in range(max_steps):
    t0 = time.time()
    last_step = (step==max_steps - 1)

    # evert once in a while, evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss/val_loss_steps
                val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"{step} val {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.6f}\n")
    
    # once in a while evaluate hellaswag:
    if (step % 250 == 0 or last_step) and not use_compile:
        num_correct_norm = 0
        num_total = 0
        from hellaswag import iterate_examples, render_example
        for i, example in enumerate(iterate_examples("val")):
            # only process assigned examples
            if i % ddp_world_size != ddp_rank:
                continue

            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"step {step}, hellaswag acc_norm {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.6f}\n")
            if step > 0 and (step % 10 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"checkpoint_{step}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),

                }
                torch.save(checkpoint, checkpoint_path)


    # once in a while. generate from the model,
    # disabled because torch.compile throws an error
    if ((step > 0 and step % 250 == 0) or last_step) and not use_compile :
        model.eval()
        num_return_sequences = 4
        max_length  =32

        tokens = enc.encode("Hello I'm a language model,")
        tokensgen = torch.tensor(tokens, dtype=torch.long)
        tokensgen = tokensgen.unsqueeze(0).repeat(num_return_sequences, 1) # (num_return_sequences, T)
        xgen = tokensgen.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42+ddp_rank)


        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # taking logits of last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get probabilties
                probs = F.softmax(logits, dim=-1) # (B, vocab_size)
                # do top-k sampling of 50 
                # topkprobs here becomes (5, 50)
                topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
                # sample from the topk probabilities
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                # gather the corresponing indices 
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1) # (B, T+1)


        # print the generated text
        for i in range(num_return_sequences):
            tokens = x[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} rample {i} >{decoded}")



    # training loop 
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # parameters will continue to be in float32, but the activations will be in bfloat16
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps # this is the loss normalized by the number of gradient accumulation steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step==grad_accum_steps-1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # global norm: sum squared gradients and take square root

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize() # wait for the gpu to finish
    t1 = time.time() 
    dt = (t1 - t0 ) # convert to milliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_Sec = tokens_processed / (t1-t0) # throughput in tokens per second
    if master_process:
        print(f"step {step}, loss {loss_accum.item():.4f} | dt: {dt:.2f} ms | norm: {norm:.2f} | Tokens per second: {   tokens_per_Sec:.2f} | tokens processed: {tokens_processed}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()

import sys
sys.exit(0)
model.eval() # set the model to evaluation mode

num_return_sequences = 5
max_length = 30



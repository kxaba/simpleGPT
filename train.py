import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
batch_size = 32
block_size = 8
max_iter = 5000
eval_interval = max_iter/10
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters= 200
n_embed = 32

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

str_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_str = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [str_to_int[c] for c in s]
decode = lambda l: ''.join(int_to_str[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            _, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
 
class Head(nn.Module):

    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        w = q @ k.transpose(-2,-1)* C**-0.5
        w = w.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)

        v = self.value(x)
        out = w @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class FeedForward(nn.Module):

    def __init__(self, n_embed) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, n_embed), nn.ReLU())
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed// n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x
    
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(Block(n_embed, n_head=4),
                                    Block(n_embed, n_head=4),
                                    Block(n_embed, n_head=4),)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tk_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))
        x = tk_embed + pos_embed
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:  
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next), dim=1)

        return idx

model = BigramLanguageModel(vocab_size).to(device)
optmizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iter):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
     
    logits, loss = model(xb,yb)
    optmizer.zero_grad(set_to_none=True)
    loss.backward()
    optmizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context , max_new_tokens=500)[0].tolist()))

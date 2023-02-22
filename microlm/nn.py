import torch


class Embedding(torch.nn.Module):

    def __init__(self, dim, num_tokens, num_positions, device, dropout=0.1, pad_ix=0):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.num_positions = num_positions
        self.device = device
        self.dropout = dropout
        self.pad_ix = pad_ix

        self.tokens = torch.nn.Embedding(self.num_tokens, self.dim, padding_idx=self.pad_ix)
        self.positions = torch.nn.Embedding(self.num_positions, self.dim)
        self.drop = torch.nn.Dropout(self.dropout)
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        xt = self.tokens(x)
        xp = torch.arange(seq_len).unsqueeze(0).tile([batch_size, 1]).long().to(self.device)
        xp = self.positions(xp)
        x = xt + xp
        x = self.drop(x)
        return x


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, num_heads, dim, device, is_causal=True):
        super().__init__()
        assert dim % num_heads == 0, "'num_heads' not divisible by 'input_dim'"
        self.num_heads = num_heads
        self.input_dim = dim
        self.is_causal = is_causal
        self.device = device
        self.hidden_dim = self.input_dim // self.num_heads

        q_size = (1, self.num_heads, self.input_dim, self.hidden_dim)
        self.q = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(q_size)))
        k_size = (1, self.num_heads, self.input_dim, self.hidden_dim)
        self.k = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(k_size)))
        v_size = (1, self.num_heads, self.input_dim, self.hidden_dim)
        self.v = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(v_size)))
        o_size = (1, self.input_dim, self.input_dim)
        self.o = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(o_size)))
        self.norm = torch.sqrt(torch.as_tensor(self.hidden_dim))

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        seq_len = q.size(1)
        # project via broadcasted 4d matmul.
        # to enable broadcasting we need to unsqueeze inputs over dim of attention heads which is dim=1 of 4d tensor.
        # that is: BxSxC -> Bx1xSxC.
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        q = torch.matmul(q, self.q)
        k = torch.matmul(k, self.k)
        v = torch.matmul(v, self.v)
        # compute attention.
        x = torch.matmul(q, k.transpose(-1, -2))
        x = x / self.norm
        # mask tokens to be ignored.
        if mask is not None:
            mask = mask.bool().to(self.device)
            mask_row = mask.view(batch_size, 1, seq_len, 1)
            mask_col = mask.view(batch_size, 1, 1, seq_len)
            x.masked_fill_(mask_row, 0.0)
            x.masked_fill_(mask_col, 0.0)
        # mask look-ahead positions when in causal mode.
        if self.is_causal:
            mask_size = (seq_len, seq_len)
            # mask out upper diagonal so token 'i' attends only to itself and tokens less than 'i'.
            mask = torch.triu(torch.ones(mask_size), diagonal=1)
            mask = mask.bool().to(self.device)
            x.masked_fill_(mask, float('-inf'))
        x = torch.softmax(x, dim=-1)
        x = torch.matmul(x, v)
        # concat all the heads back to 3d tensor of the same shape as input.
        # that is: BxAxSxH -> BxSxA*H -> BxSxC.
        x = x.view(batch_size, seq_len, self.num_heads * self.hidden_dim)
        x = torch.matmul(x, self.o)
        return x


class FFN(torch.nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.input_dim = dim
        self.hidden_dim = hidden_dim
        
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.input_dim),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class EncoderBlock(torch.nn.Module):
    
    def __init__(self, num_heads, dim, ffn_dim, device, dropout=0.1, is_causal=True):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = dim
        self.ffn_dim = ffn_dim
        self.device = device
        self.dropout = dropout
        self.is_causal = is_causal

        self.attn = MultiHeadAttention(num_heads=self.num_heads, dim=self.input_dim, is_causal=self.is_causal, device=self.device)
        self.attn_norm = torch.nn.LayerNorm(self.input_dim)
        self.ffn = FFN(dim=self.input_dim, hidden_dim=self.ffn_dim)
        self.ffn_norm = torch.nn.LayerNorm(self.input_dim)
        self.drop = torch.nn.Dropout(self.dropout)

    def forward(self, x, mask=None):
        x_attn = self.attn(q=x, k=x, v=x, mask=mask)
        x_attn = self.drop(x_attn)
        x = x + x_attn
        x = self.attn_norm(x)
        x_ffn = self.ffn(x)
        x_ffn = self.drop(x_ffn)
        x = x + x_ffn
        x = self.ffn_norm(x)
        return x


class Encoder(torch.nn.Module):
    
    def __init__(self, num_blocks, num_heads, dim, ffn_dim, device, dropout=0.1, is_causal=True):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.device = device
        self.is_causal = is_causal
        self.dropout = dropout

        self.blocks = torch.nn.ModuleList([
            EncoderBlock(
                num_heads=self.num_heads,
                dim=self.dim,
                ffn_dim=self.ffn_dim,
                dropout=self.dropout,
                is_causal=self.is_causal,
                device=self.device
            )
            for _ in range(self.num_blocks)
        ])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask=mask)
        return x


class Head(torch.nn.Module):
    
    def __init__(self, dim, num_tokens):
        super().__init__()
        self.input_dim = dim
        self.num_tokens = num_tokens

        self.layer = torch.nn.Linear(self.input_dim, self.num_tokens)
    
    def forward(self, x):
        x = x.mean(dim=1)
        x = self.layer(x)
        return x


class Transformer(torch.nn.Module):

    @classmethod
    def from_config(cls, config, tokenizer):
        device = config['model']['device']
        model = Transformer(
            embedding=Embedding(
                dim=config['model']['dim'],
                num_tokens=tokenizer.num_chars,
                num_positions=tokenizer.seq_len,
                device=device
            ),
            encoder=Encoder(
                num_blocks=config['model']['blocks'],
                num_heads=config['model']['heads'],
                dim=config['model']['dim'],
                ffn_dim=config['model']['ffn'],
                dropout=config['model']['dropout'],
                is_causal=True,
                device=device
            ),
            head=Head(
                dim=config['model']['dim'],
                num_tokens=tokenizer.num_chars
            ),
        )
        return model

    def __init__(self, embedding, encoder, head):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.head = head
    
    def forward(self, ids, mask=None):
        x = self.embedding(ids)
        x = self.encoder(x, mask=mask)
        x = self.head(x)
        return x


class LSTM(torch.nn.Module):

    @classmethod
    def from_config(cls, config, tokenizer):
        model = cls(num_tokens=tokenizer.num_chars)
        return model

    def __init__(self, num_tokens, hidden_size=512):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(self.num_tokens, self.hidden_size, padding_idx=0)
        self.lstm = torch.nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.head = torch.nn.Linear(self.hidden_size, self.num_tokens)
    
    def forward(self, ids, mask=None):
        x = self.embedding(ids)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

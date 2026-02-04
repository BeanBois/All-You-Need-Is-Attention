import torch 
import torch.nn.functional as F


# Scaled Dot-Product Attention
class AttentionBlock(torch.nn.Module):
    def __init__(self, dk, dv):
        super().__init__()
        self.dk = dk
        self.dv = dv
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, queries, keys, values, mask = None):
        # first compute the dot product of queries and keys 
        dp = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float32))

        # apply mask if provided
        if mask is not None:
            dp = dp + mask

        # apply softmax to get attention weights
        attn_weights = self.softmax(dp)

        # compute the weighted sum of values
        output = torch.matmul(attn_weights, values)
        return output, attn_weights
    
class MultiHeadAttentionBlock(torch.nn.Module):

    def __init__(self, num_heads, dk, dv):
        super().__init__()
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv 
        self.d_model = self.dk * self.num_heads

        self.W_q = torch.nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.W_k = torch.nn.Parameter(torch.randn(self.d_model , self.d_model))
        self.W_v = torch.nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.W_o = torch.nn.Parameter(torch.randn(self.d_model , self.d_model))

        self.attention_block = AttentionBlock(dk, dv)

    def forward(self, queries, keys, values, mask = None):

        batch_size = queries.size(0)

        # Linear projections

        Q = torch.matmul(queries, self.W_q) # (B, seq, d_model)
        # Q = [ input@W_q_1 | input@W_q_2 | ... | input@W_q_8 ]
        #       (64 dims)     (64 dims)          (64 dims)
        #     |______________________________________________|
        #                      512 dims
        K = torch.matmul(keys, self.W_k)
        V = torch.matmul(values, self.W_v)

        # Split into multiple heads
        # without transposing, you'd be computing attention across heads, 
        # which is wrong. You want attention across sequence positions.
        Q = Q.view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2) # (B, seq, num_heads, dk) => (B, num_heads, seq, dk)
        K = K.view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.dv).transpose(1, 2)

        # Apply attention on each head
        # attn_outputs: (B, num_heads, seq, dv)
        attn_outputs, attn_weights = self.attention_block.forward(Q, K, V, mask)

        # Concatenate heads
        # first undo transposing : (B, num_heads, seq, dv) => (B, seq, num_heads, dv)
        # then call contiguous to make sure the tensor is stored in a contiguous chunk of memory
        # finally, reshape to (B, seq, num_heads * dv)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # (B, seq, num_heads * dv)

        # Final linear layer
        # (batch, seq, 512) @ (512, 512) → (batch, seq, 512)
        output = torch.matmul(attn_outputs, self.W_o)

        return output, attn_weights

# Position-wise Feed-Forward Network
class FeedForwardBlock(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.randn(d_model, d_ff))
        self.b1 = torch.nn.Parameter(torch.randn(d_ff))
        self.W2 = torch.nn.Parameter(torch.randn(d_ff, d_model))
        self.b2 = torch.nn.Parameter(torch.randn(d_model))

    def forward(self, x):
        # x: (batch, seq, d_model)
        x = torch.matmul(x, self.W1) + self.b1  # (batch, seq, d_ff)
        x = F.relu(x)
        x = torch.matmul(x, self.W2) + self.b2  # (batch, seq, d_model)
        return x

def positional_encoder(seq_len, dim):
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(0, dim, 2)
    angles = pos / (10000 ** (2 * i / dim))
    
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe

# torch.nn.Embedding for learned positional encodings
class LearnedPositionalEncodings(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()  
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim))
    
    def forward(self, indices):
        return self.weight[indices]

def create_causal_mask(seq_len):
    # Upper triangle (above diagonal) = True
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    # Convert: True positions become -inf, False become 0
    return mask.float().masked_fill(mask, float('-inf'))
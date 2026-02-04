from utils import MultiHeadAttentionBlock, FeedForwardBlock, positional_encoder, create_causal_mask, LearnedPositionalEncodings
from torch.nn import LayerNorm, ModuleList, Module
from encoder import Encoder



class DecoderLayer(Module):

    def __init__(self, output_dim, num_heads,):
        super().__init__()  
        self.mha = MultiHeadAttentionBlock(num_heads=num_heads, dk=output_dim//num_heads, dv=output_dim//num_heads)
        self.cross_mha = MultiHeadAttentionBlock(num_heads=num_heads, dk=output_dim//num_heads, dv=output_dim//num_heads)
        self.ffn = FeedForwardBlock(d_model=output_dim, d_ff=output_dim*4)
        self.layernorm1 = LayerNorm(output_dim)
        self.layernorm2 = LayerNorm(output_dim)
        self.layernorm3 = LayerNorm(output_dim)
        self.register_buffer('mask', create_causal_mask(1000))

    def forward(self, x, encoder_output):
        
        x1, _ = self.mha.forward(x, x, x, mask=self.mask[:x.size(1), :x.size(1)])
        x1 = self.layernorm1(x1 + x)
        x2, _ = self.cross_mha.forward(x1, encoder_output, encoder_output)
        x2 = self.layernorm2(x2 + x1)
        x3 = self.ffn.forward(x2)        
        x3 = self.layernorm3(x3 + x2)
        return x3


class Decoder(Module):

    def __init__(self, vocab_size, num_layers = 6, output_dim=512, num_heads=8):
        super().__init__()
        self.embedding = LearnedPositionalEncodings(vocab_size, output_dim)
        self.layers = [DecoderLayer(output_dim=output_dim, num_heads=num_heads) for _ in range(num_layers)]
        self.layers = ModuleList(self.layers)
        self.num_layers = num_layers    

    def forward(self, x, encoder_output):
        x = self.embedding(x)
        x = x + positional_encoder(x.size(1), x.size(2)).to(x.device)
        for i in range(self.num_layers):
            x = self.layers[i].forward(x, encoder_output)
        return x
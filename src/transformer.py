import torch
from .decoder import Decoder
from .encoder import Encoder

class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, num_layers=6, output_dim=512, num_heads=8):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size=src_vocab_size, num_layers=num_layers, output_dim=output_dim, num_heads=num_heads)
        self.decoder = Decoder(vocab_size=tgt_vocab_size, num_layers=num_layers, output_dim=output_dim, num_heads=num_heads)
        self.output_linear = torch.nn.Linear(output_dim, tgt_vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, src, tgt, train=True):
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt, encoder_output)
        output = self.output_linear(decoder_output)
        if train:
            return output
        output = self.softmax(output)
        return output
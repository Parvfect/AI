"""
https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
"""

import torch
import math

class Embedder(nn.Module):
    """ The Embedding module that returns the embedded words using the PyTorch embedding module when the orignal 
    sequences are passed to it """

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # Initializes self.embed as the embedding function based on the vocab_size and encoding_dim
        self.embed = nn.Embedding(vocab_size, encoding_dim)

    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    """ A constant abstract vector that holds information of the position of each word in the sequence
    based on its position in the sequence and in the model dimension, that gets added to the encoding values
    to make sure that the model also has information about the context """

    def __init__(self, encoding_dim, max_seq_len = 80):
        super().__init__()
        self.encoding_dim = encoding_dim

        # pe - Positional Encoding Vector
        pe = torch.zeros(self.encoding_dim, max_seq_len)

        for pos in range(max_seq_len):
            for i in range(0, self.encoding_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] =  math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        # Increases dimensionality so that orignal meaning of positional encoding is not lost when they are added together
        pe = pe.unsqueeze(0)

        # Makes sure the weights aren't trained by the optimizers
        self.register_buffer('pe', pe)

    def forward(self, x):

        # Make embeddings larger
        x = x * math.sqrt(self.encoding_dim)

        seq_len = x.size(1)

        # Add constant to embedding
        x = x + Variable(self.pe[:,:seq_len], requires_grad = False).cuda()

        return x

""" Creating Masks """

""" Multi Head Attention Layer """

""" Feed Forward Layer """

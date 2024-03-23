from datasets import load_dataset
from torch import nn
from torch.nn import functional as F
import torch

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
import torchdata


# import datasets
from torchtext.datasets import IMDB

# train_iter = IMDB(split='train')

# def tokenize(label, line):
#     return line.split()

# tokens = []
# for label, line in train_iter:
#     tokens += tokenize(label, line)

# from torchtext.vocab import build_vocab_from_iterator
# from collections import Counter
# counter = Counter(tokens)
# vocab = sorted(counter, key=counter.get, reverse=True)
# int2word = dict(enumerate(vocab, 1))
# int2word[0] = '<PAD>'
# word2int = {word: id for id, word in int2word.items()}

# # Load the IMDb dataset
# def load_imdb_data():
#     # Assuming the dataset is downloaded and extracted in the current directory
#     data_dir = "./aclImdb"
#     train_dir = os.path.join(data_dir, "train")
    
#     # Load data and labels
#     raw_data = load_files(train_dir, categories=['pos', 'neg'])
#     data, labels = raw_data.data, raw_data.target

#     # Split data into train and validation sets
#     train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

#     return train_data, val_data, train_labels, val_labels

# # Step 2: Custom Dataset Class
# class IMDbDataset(Dataset):
#     def __init__(self, data, labels, transform=None):
#         self.data = data
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = {'data': self.data[idx], 'label': self.labels[idx]}
        
#         if self.transform:
#             sample = self.transform(sample)
            
#         return sample

# # Step 3: Implement Iterator
# class IMDbDataIterator:
#     def __init__(self, dataset, batch_size=32, shuffle=True):
#         self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#     def __iter__(self):
#         for batch in self.dataloader:
#             yield batch

# # Define your transformations if needed
# transform = transforms.Compose([
#     # Your transformations here
# ])

# # Load IMDb data
# train_data, val_data, train_labels, val_labels = load_imdb_data()

# # Create the dataset
# train_dataset = IMDbDataset(train_data, train_labels, transform=transform)
# val_dataset = IMDbDataset(val_data, val_labels, transform=transform)

# # Create the iterators
# train_iterator = IMDbDataIterator(train_dataset, batch_size=32)
# val_iterator = IMDbDataIterator(val_dataset, batch_size=32)

# Example usage:
# for batch in train_iterator:
#     # Perform training using the batch
#     data, labels = batch['data'], batch['label']
#     # Forward pass, backward pass, update weights, etc.
#
# for batch in val_iterator:
#     # Perform validation using the batch
#     data, labels = batch['data'], batch['label']
#     # Calculate validation metrics, etc.


import random, tqdm, sys, math, gzip
def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """

    h, w = matrices.size(-2), matrices.size(-1)

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[..., indices[0], indices[1]] = maskval


class SelfAttentionGPT2(nn.Module):
    """
    This is the self-attention operation as implemented in the Huggingface port of GPT2. The code has been
    simplified to remove several features not used here but otherwise it should do exactly the same as GPT2 when run with
    normal parameters.

    It is very similar to the default SelfAttention below, with the exception of the way it's initialized and some
    small speed improvements in the custom implementation of the linear layer (the Conv1D defined above).

    We include this primarily for comparison with our own canonical implementation to check for performance differences.
    """
    def __init__(self, emb, heads, mask=False):
        super().__init__()

        self.nheads = heads
        self.emb = emb
        self.mask = mask

        #self.c_attn = Conv1D(3 * emb, emb)
        # -- (out_channels, in_channels):
        #    This is a very slight modification of a linear layer

        self.c_attn = nn.Linear(emb, 3*emb)
        #self.c_proj = Conv1D(emb, emb)
        self.c_proj = nn.Linear(emb, emb)

    def _attn(self, q, k, v):
        dot = torch.matmul(q, k) # raw attention weights
        dot = dot / (float(v.size(-1)) ** 0.5) # scaled attention weights
        if self.mask: # Apply the attention mask
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        # -- This is implemented differently in the Huggingface version, but the effect should be the same.
        dot = nn.Softmax(dim=-1)(dot) # normalized attention weights
        return torch.matmul(dot, v) # attention over values

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, is_key=False):
        new_x_shape = x.size()[:-1] + (self.nheads, x.size(-1) // self.nheads)
        x = x.view(*new_x_shape)
        if is_key:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, input_sequence):
        b, t, e = input_sequence.size()
        query, key, value = self.c_attn(input_sequence).split(e, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, is_key=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a



class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask=False):
        super().__init__()
        assert k % heads == 0
        self.k, self.heads = k, heads

        self.tokeys = nn.Linear(k, k, bias=False)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tovalues = nn.Linear(k, k, bias=False)

        self.unifyheads = nn.Linear(k, k)

    def forward(self, x):
        b,t,k = x.size()
        h = self.heads
        queries = self.toqueries(x)
        keys = self.tokeys(x)
        values = self.tovalues(x)

        s = k // h
        queries = queries.view(b, t, h, s)
        keys = keys.view(b, t, h, s)
        values = values.view(b, t, h, s)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)
        
        # Get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # -- dot has size (b*h, t, t) containing raw weights

        # scale the dot product
        dot = dot / (k ** (1/2))
        
        # normalize 
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
    
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    """
    A straightforward transformer block.
    """

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, attention_type=SelfAttention,
                 pos_embedding=None, sa_kwargs={}):
        super().__init__()

        try:
            self.attention = attention_type(emb, heads=heads, mask=mask, **sa_kwargs)
        except:
            raise Exception(f'Self-attention type {type} not recognized.')
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(

            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.do(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.do(x)
        return x

class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0,  sa_module = SelfAttentionGPT2, wide=False):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_tokens: Number of tokens (usually words) in the vocabulary
        :param num_classes: Number of classes.
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()

        self.num_tokens, self.max_pool = num_tokens, max_pool
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)
        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, attention_type=sa_module, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(emb, num_classes)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()
        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.do(x)
        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)




def go(arg):
    """
    Creates and trains a basic transformer for the IMDB sentiment classification task.
    """
    model = CTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=arg.max_length, num_tokens=arg.vocab_size, num_classes=2, max_pool=arg.max_pool)

    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')

    def tokenize(label, line):
        return line.split()

    tokens = []
    num_samples = 0
    for label, line in train_iter:
        tokens += tokenize(label, line)
        num_samples += 1

    for label,line in test_iter:
        tokens += tokenize(label, line)

    from collections import Counter
    counter = Counter(tokens)
    vocab = sorted(counter, key=counter.get, reverse=True)
    int2word = dict(enumerate(vocab, 1))
    int2word[0] = '<PAD>'
    word2int = {word: id for id, word in int2word.items()}

    # id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    # label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))


    shuffler = torchdata.datapipes.iter.Shuffler(train_iter, buffer_size= 25000)

    # training loop
    seen = 0
    for e in range(arg.num_epochs):

        print(f'\n epoch {e}')
        model.train(True)

        with tqdm.tqdm(total=num_samples//arg.batch_size, smoothing=0., position=0, leave=True) as pbar:
            for batch in shuffler.batch(batch_size=arg.batch_size):
                opt.zero_grad()
                tokenized_sequences = [torch.tensor([word2int[token] for token in tokenize(sample[0], sample[1])], dtype=torch.int64) for sample in batch]
                input = pad_sequence(tokenized_sequences, batch_first=True)
                label = torch.tensor([sample[0] -1 for sample in batch], dtype=torch.int64)

                if input.shape[1] > arg.max_length:
                    input = input[:, :arg.max_length]
                out = model(input)
                loss = F.nll_loss(out, label)

                loss.backward()
                print(f"Loss: {loss}")

                # clip gradients
                # - If the total gradient vector has a length > 1, we clip it back down to 1.
                if arg.gradient_clipping > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

                opt.step()
                sch.step()

                seen += input.size(0)
                tbw.add_scalar('classification/train-loss', float(loss.item()), seen)
                pbar.update(1)

        with torch.no_grad():

            model.train(False)
            tot, cor= 0.0, 0.0

            for batch in test_iter.batch(batch_size=arg.batch_size):


                tokenized_sequences = [torch.tensor([word2int[token] for token in tokenize(sample[0], sample[1])], dtype=torch.int64) for sample in batch]
                input = pad_sequence(tokenized_sequences, batch_first=True)
                label = torch.tensor([sample[0] -1 for sample in batch], dtype=torch.int64)

                if input.size(1) > arg.max_length:
                    input = input[:, :arg.max_length]
                out = model(input).argmax(dim=1)

                tot += float(input.size(0))
                cor += float((label == out).sum().item())

            acc = cor / tot
            print(f'-- {"test" if arg.final else "validation"} accuracy {acc:.3}')
            tbw.add_scalar('classification/test-loss', float(loss.item()), e)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=1, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=32, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=300000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10_000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
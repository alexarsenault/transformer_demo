from torch import nn
from torch.nn import functional as F
import torch
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import tqdm, math


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

        # self.c_attn = Conv1D(3 * emb, emb)
        # -- (out_channels, in_channels):
        #    This is a very slight modification of a linear layer

        self.c_attn = nn.Linear(emb, 3 * emb)
        # self.c_proj = Conv1D(emb, emb)
        self.c_proj = nn.Linear(emb, emb)

    def _attn(self, q, k, v):
        dot = torch.matmul(q, k)  # raw attention weights
        dot = dot / (float(v.size(-1)) ** 0.5)  # scaled attention weights
        if self.mask:  # Apply the attention mask
            mask_(dot, maskval=float("-inf"), mask_diagonal=False)
        # -- This is implemented differently in the Huggingface version, but the effect should be the same.
        dot = nn.Softmax(dim=-1)(dot)  # normalized attention weights
        return torch.matmul(dot, v)  # attention over values

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
        b, t, k = x.size()
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
        dot = dot / (k ** (1 / 2))

        # normalize
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    """
    A straightforward transformer block.
    """

    def __init__(
        self,
        emb,
        heads,
        mask,
        seq_length,
        ff_hidden_mult=4,
        dropout=0.1,
        attention_type=SelfAttention,
        pos_embedding=None,
        sa_kwargs={},
    ):
        super().__init__()

        try:
            self.attention = attention_type(emb, heads=heads, mask=mask, **sa_kwargs)
        except:
            raise Exception(f"Self-attention type {type} not recognized.")
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb),
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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(
        self,
        emb,
        heads,
        depth,
        seq_length,
        num_tokens,
        num_classes,
        max_pool=True,
        dropout=0.0,
        sa_module=SelfAttentionGPT2,
        wide=False,
    ):
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
        self.value_embedding = nn.Linear(1, emb, dtype=torch.double)
        # self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)
        self.pos_encoding = PositionalEncoding(emb, max_len=seq_length, dropout=0.2)
        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(
                    emb=emb,
                    heads=heads,
                    seq_length=seq_length,
                    mask=False,
                    attention_type=sa_module,
                    dropout=dropout,
                )
            )

        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(emb, num_classes)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        x = self.value_embedding(x)
        x = self.pos_encoding(x)
        x = x.to(dtype=torch.float)
        x = self.tblocks(x)
        x = (
            x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)
        )  # pool over the time dimension
        x = self.toprobs(x)
        return F.log_softmax(x, dim=1)


def main(arg):
    """
    Creates and trains a basic transformer for the IMDB sentiment classification task.
    """
    model = CTransformer(
        emb=arg.embedding_size,
        heads=arg.num_heads,
        depth=arg.depth,
        seq_length=arg.max_length,
        num_tokens=arg.vocab_size,
        num_classes=4,
        max_pool=arg.max_pool,
    )
    model = model.to("cuda")

    from tslearn.datasets import CachedDatasets

    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    num_x_samples = len(X_train)
    num_y_samples = len(y_train)
    tbw = SummaryWriter(log_dir=arg.tb_dir)  # Tensorboard logging

    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0)
    )

    # training loop
    seen = 0
    for e in range(arg.num_epochs):

        print(f"\n epoch {e}")
        model.train(True)

        with tqdm.tqdm(
            total=num_x_samples, smoothing=0.0, position=0, leave=True
        ) as pbar:
            for x_samp, label in zip(X_train, y_train):
                opt.zero_grad()

                x_samp = (
                    torch.tensor(x_samp).to("cuda").squeeze().unsqueeze(0).unsqueeze(-1)
                )
                label = torch.tensor(label - 1).to("cuda").unsqueeze(0)

                out = model(x_samp)
                loss = F.nll_loss(out, label)

                loss.backward()

                # clip gradients
                # - If the total gradient vector has a length > 1, we clip it back down to 1.
                if arg.gradient_clipping > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

                opt.step()
                sch.step()

                seen += x_samp.shape[0]
                tbw.add_scalar("classification/train-loss", float(loss.item()), seen)
                pbar.update(1)

        with torch.no_grad():

            model.train(False)
            tot, cor = 0.0, 0.0

            for x_samp, label in zip(X_test, y_test):

                x_samp = (
                    torch.tensor(x_samp).to("cuda").squeeze().unsqueeze(0).unsqueeze(-1)
                )
                label = torch.tensor(label - 1).to("cuda").unsqueeze(0)

                out = model(x_samp)
                pred = out.argmax(dim=1)

                loss = F.nll_loss(out.to(torch.float32), label.to(torch.int64))

                tot += float(x_samp.shape[0])
                cor += float((label == pred).sum().item())

            acc = cor / tot
            print(f'-- {"test" if arg.final else "validation"} accuracy {acc:.3}')
            tbw.add_scalar("classification/test-loss", float(loss.item()), e)
            tbw.add_scalar("classification/test-acc", acc, e)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "-e",
        "--num-epochs",
        dest="num_epochs",
        help="Number of epochs.",
        default=100,
        type=int,
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        help="The batch size.",
        default=4,
        type=int,
    )

    parser.add_argument(
        "-l",
        "--learn-rate",
        dest="lr",
        help="Learning rate",
        default=0.0001,
        type=float,
    )

    parser.add_argument(
        "-T",
        "--tb_dir",
        dest="tb_dir",
        help="Tensorboard logging directory",
        default="./runs",
    )

    parser.add_argument(
        "-f",
        "--final",
        dest="final",
        help="Whether to run on the real test set (if not included, the validation set is used).",
        action="store_true",
    )

    parser.add_argument(
        "--max-pool",
        dest="max_pool",
        help="Use max pooling in the final classification layer.",
        action="store_true",
    )

    parser.add_argument(
        "-E",
        "--embedding",
        dest="embedding_size",
        help="Size of the character embeddings.",
        default=128,
        type=int,
    )

    parser.add_argument(
        "-V",
        "--vocab-size",
        dest="vocab_size",
        help="Number of words in the vocabulary.",
        default=438730,
        type=int,
    )

    parser.add_argument(
        "-M",
        "--max",
        dest="max_length",
        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
        default=275,
        type=int,
    )

    parser.add_argument(
        "-H",
        "--heads",
        dest="num_heads",
        help="Number of attention heads.",
        default=8,
        type=int,
    )

    parser.add_argument(
        "-d",
        "--depth",
        dest="depth",
        help="Depth of the network (nr. of self-attention layers)",
        default=4,
        type=int,
    )

    parser.add_argument(
        "-r",
        "--random-seed",
        dest="seed",
        help="RNG seed. Negative for random",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--lr-warmup",
        dest="lr_warmup",
        help="Learning rate warmup.",
        default=10_000,
        type=int,
    )

    parser.add_argument(
        "--gradient-clipping",
        dest="gradient_clipping",
        help="Gradient clipping.",
        default=1.0,
        type=float,
    )

    options = parser.parse_args()

    print("OPTIONS ", options)

    main(options)

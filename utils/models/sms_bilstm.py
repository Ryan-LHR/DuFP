import torch
import torch.nn as nn
import torch.nn.functional as F

class Representation(nn.Module):
    def __init__(self, hidden_dim):
        super(Representation, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, lstm_out):
        hidden_out = torch.cat(
            (lstm_out[:, -1, :self.hidden_dim],
             lstm_out[:, 0, self.hidden_dim:]),
            dim=1
        )
        return hidden_out

class SentimentRNN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, output_dim, embedding_dim, drop_prob=0.5):
        super(SentimentRNN, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=no_layers,
            dropout=drop_prob,
            bidirectional=True,
            batch_first=True
        )

        self.representation = Representation(hidden_dim)

        # dropout layer
        # self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim * 2, output_dim)
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # hidden_out = torch.cat((lstm_out[:, -1, :self.hidden_dim], lstm_out[:, 0, self.hidden_dim:]), dim=1)
        hidden_out = self.representation(lstm_out)
        out = self.fc(hidden_out)
        # sig_out = self.sig(out)
        sig_out = self.softmax(out)
        if len(sig_out.shape) == 1:
            sig_out = sig_out.view(-1, sig_out.size(0))
        return sig_out

    def forward_feature(self, x, pad_idx=0):
        token_ids = x  # [DIFF 1] keep token ids for mask

        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)  # (B, L, 2H)

        mask = (token_ids != pad_idx)  # [DIFF 2] padding mask: (B, L)
        lengths = mask.sum(dim=1)      # (B,)

        B, L, _ = lstm_out.size()
        batch_idx = torch.arange(B, device=lstm_out.device)

        # first valid index (left-pad safe; also works generally)
        first_idx = mask.int().argmax(dim=1)
        # last valid index (robust: find from the end)
        last_from_end = mask.flip(dims=[1]).int().argmax(dim=1)
        last_idx = (L - 1) - last_from_end

        # handle empty sequences (just in case)
        has_token = lengths > 0
        first_idx = torch.where(has_token, first_idx, torch.zeros_like(first_idx))
        last_idx = torch.where(has_token, last_idx, torch.zeros_like(last_idx))

        # [DIFF 3] gather forward(last valid) and backward(first valid)
        fwd = lstm_out[batch_idx, last_idx, :self.hidden_dim]
        bwd = lstm_out[batch_idx, first_idx, self.hidden_dim:]
        hidden_out = torch.cat([fwd, bwd], dim=1)

        return hidden_out  # [DIFF 4] return feature directly for DiMP

def SMS_BiLSTM():
    """The official model define from TDPR"""
    return SentimentRNN(2, 1001, 64, 2, 64, 0.5)

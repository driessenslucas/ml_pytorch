import torch
import torch.nn as nn
from torchtext.datasets import IMDB

train_dataset = IMDB(split="train")
test_dataset = IMDB(split="test")

# step1: create the datasets
from torch.utils.data.dataset import random_split

torch.manual_seed(1)
train_dataset, valid_dataset = random_split(list(train_dataset), 
                                        [20000, 5000])

# step2: find unique tokens
import re
from collections import Counter, OrderedDict

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
    + ' '.join(emoticons).replace('-', '')
    tokenized = text.split()
    return tokenized

token_counts = Counter()
for label, line in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)

#print("Vocabulary size:", len(token_counts))

# Step3: encoding each unique token into integers
from torchtext.vocab import vocab
sorted_by_frew_tuples = sorted(
    token_counts.items(), key=lambda x: x[1], reverse=True
)
ordered_dict = OrderedDict(sorted_by_frew_tuples)
vocab = vocab(ordered_dict)
vocab.insert_token("<pad>", 0)
vocab.insert_token("<unk>", 1)
vocab.set_default_index(1)

#print([vocab[token] for token in ['this', 'is', 'an', 'example']])

# Define the functions for transformation
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1. if x == 'pos' else 0.

## Wrap the encode and transformation function

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text),
                            dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))

    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(
        text_list, batch_first=True
    )
    return padded_text_list, label_list, lengths

# ## Take a s small abtch
from torch.utils.data import DataLoader
# dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False,
#                         collate_fn=collate_batch)
# text_batch, label_batch, length_batch = next(iter(dataloader))
#print(text_batch)

## Dataloaders
batch_size = 32
train_dl = DataLoader(train_dataset, batch_size=batch_size,
                     shuffle=True, collate_fn=collate_batch)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size,
                     shuffle=True, collate_fn=collate_batch)
test_dl = DataLoader(test_dataset, batch_size=batch_size,
                     shuffle=True, collate_fn=collate_batch)

## Embedding
embedding = nn.Embedding(
    num_embeddings=10,
    embedding_dim=3,
    padding_idx=0)


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size,
               fc_hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size,
                        batch_first=True)
        self.embedding = nn.Embedding(vocab_size,
                                    embed_dim,
                                    padding_idx=0)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(
            out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True
        )
        out, (hidden, cell) = self.rnn(out)
        out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

vocab_size = len(vocab)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64
torch.manual_seed(1)
model = RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)


# train
def train(dataloader):
    model.train()
    total_acc, total_loss = 0, 0








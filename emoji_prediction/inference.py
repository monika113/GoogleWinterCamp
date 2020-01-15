import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import spacy
import os
cur_dir = os.path.abspath(os.path.dirname(__file__))
import sys
sys.path.append(cur_dir)
import pickle
import nltk
from nltk import word_tokenize
nltk.download('punkt')

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
    
    def forward(self, encoder_outputs):
        # encoder_outputs = [batch size, sent len, hid dim]
        energy = self.projection(encoder_outputs)
        # energy = [batch size, sent len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # weights = [batch size, sent len]
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        # outputs = [batch size, hid dim]
        return outputs, weights


class AttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, dropout=dropout)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x = [sent len, batch size]
        embedded = self.embedding(x)
        # embedded = [sent len, batch size, emb dim]
        output, (hidden, cell) = self.lstm(embedded)
        # use 'batch_first' if you want batch size to be the 1st para
        # output = [sent len, batch size, hid dim*num directions]
        output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        # output = [sent len, batch size, hid dim]
        ouput = output.permute(1, 0, 2)
        # ouput = [batch size, sent len, hid dim]
        new_embed, weights = self.attention(ouput)
        # new_embed = [batch size, hid dim]
        # weights = [batch size, sent len]
        new_embed = self.dropout(new_embed)
        return self.fc(new_embed)


class Emojibot:
    def __init__(self):
        self.TEXT = data.Field(tokenize=word_tokenize)
        self.LABEL = data.LabelField()
        self.model = None
        self.device = torch.device('cuda')

    def load_model(self):
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        train_data, valid_data = data.TabularDataset.splits(
            path=cur_dir + '/data', train='train_c.csv',
            validation='dev_c.csv',
            format='csv', skip_header=True,
            csv_reader_params={'delimiter': '\t'},
            fields=[('text', self.TEXT), ('label', self.LABEL)])

        print('train_data[0]', vars(train_data[0]))

        self.TEXT.build_vocab(train_data, vectors='glove.42B.300d')
        self.LABEL.build_vocab(train_data)

        print(self.LABEL.vocab.stoi)

        BATCH_SIZE = 30
        INPUT_DIM = len(self.TEXT.vocab)
        EMBEDDING_DIM = 300
        HIDDEN_DIM = 256
        OUTPUT_DIM = len(self.LABEL.vocab)
        N_LAYERS = 2
        BIDIRECTIONAL = True
        DROPOUT = 0.5

        self.model = AttentionLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
        pretrained_embeddings = self.TEXT.vocab.vectors
        self.model.embedding.weight.data.copy_(pretrained_embeddings)

        self.model = self.model.to(self.device)

        SAVE_DIR = cur_dir + '/models'
        MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'emoji_classification_model.pt')

        self.model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        self.model.eval()

    def predict_class(self, sentence):
        tokenized = word_tokenize(sentence.lower())
        indexed = [self.TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(self.device)
        tensor = tensor.unsqueeze(1)
        preds = self.model(tensor)
        max_preds = preds.argmax(dim=1)
        return self.LABEL.vocab.itos[max_preds.item()]

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
        self.emoji_dict = {':face_with_tears_of_joy:': "\U0001F602", ':weary_face:': "\U0001F629",
':purple_heart:': "\U0001F496", ':party_popper:': "\U0001F389", ':speaking_head:': "\U0001F5E3",
':sparkles:': "\U00002728", ':loudly_crying_face:': "\U0001F62D",
':smiling_face_with_heart-eyes:': "\U0001F60A", ':person_shrugging:': "\U0001F937",
':fire:': "\U0001F525", ':person_facepalming:': "\U0001F926", ':red_heart:': "\U0001F496",
':hundred_points:': "\U0001F4AF", ':raising_hands:': "\U0001F64B", ':trophy:': "\U0001F3C6",
':beaming_face_with_smiling_eyes:': "\U0001F601", ':two_hearts:': "\U0001F495", ':heart_suit:': "\U0001F496",
':skull:': "\U0001F480", ':thumbs_up:': "\U0001F44D", ':folded_hands:': "\U0001F64F", ':flexed_biceps:': "\U0001F4AA",
':face_blowing_a_kiss:': "\U0001F618", ':smiling_face:': "\U0001F603", ':face_with_rolling_eyes:': "\U0001F644",
':crying_face:': "\U0001F622", ':OK_hand:': "\U0001F44C", ':blue_heart:': "\U0001F496", ':winking_face:': "\U0001F609",
':flushed_face:': "\U0001F633", ':clapping_hands:': "\U0001F44F", ':white_heavy_check_mark:': "\U00002705",
':smiling_face_with_sunglasses:': "\U0001F60E", ':male_sign:': "\U00002642", ':double_exclamation_mark:': "\U0000203C",
':smiling_face_with_smiling_eyes:': "\U0001F60A", ':thinking_face:': "\U0001F914",
':police_car_light:': "\U0001F6A8", ':collision:': "\U0001F4A5",
':rolling_on_the_floor_laughing:': "\U0001F923",  ':yellow_heart:': "\U0001F496", ':eyes:': "\U0001F440",
':sparkling_heart:': "\U0001F496", ':glowing_star:': "\U0001F31F", ':female_sign:': "\U00002640", ':heavy_check_mark:':"\U00002705"}

    def load_model(self):
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        train_data, valid_data = data.TabularDataset.splits(
            path=cur_dir + '/data', train='train_c.csv',
            validation='dev_c.csv',
            format='csv', skip_header=True,
            csv_reader_params={'delimiter': '\t'},
            fields=[('text', self.TEXT), ('label', self.LABEL)])

        # print('train_data[0]', vars(train_data[0]))

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
        emoji_name = self.LABEL.vocab.itos[max_preds.item()]
        emoji = self.emoji_dict[emoji_name]
        return emoji

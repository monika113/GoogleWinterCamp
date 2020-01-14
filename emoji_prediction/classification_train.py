import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import spacy
import os
import pickle
from nltk import word_tokenize

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


TEXT = data.Field(tokenize=word_tokenize)
LABEL = data.LabelField()

train_data, valid_data = data.TabularDataset.splits(
path='data', train='train_c.csv',
validation='dev_c.csv',
format='csv', skip_header=True,
csv_reader_params={'delimiter':'\t'},
fields=[('text',TEXT),('label',LABEL)])



print('train_data[0]', vars(train_data[0]))

TEXT.build_vocab(train_data, vectors='glove.42B.300d')
LABEL.build_vocab(train_data)

print(LABEL.vocab.stoi)

device = torch.device('cuda')
BATCH_SIZE = 30
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = len(LABEL.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = AttentionLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    device=device
)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)


def topk_categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    correct_cnt = 0
    for i in range(preds.size()[0]):
        topk_preds = preds[i].topk(5)[1].tolist() # get the index of the max probability
        correct = y[i].item()
        if correct in topk_preds:
            correct_cnt += 1
    return float(correct_cnt)/y.shape[0]

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum()/torch.FloatTensor([y.shape[0]])

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text)
        loss = criterion(predictions, batch.label)
        # print(predictions.size())
        # print(batch.label.size())
        acc = categorical_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)
            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 0
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'emoji_classification_model.pt')

best_valid_acc = 0
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# model.load_state_dict(torch.load(MODEL_SAVE_PATH))
# epoch_acc = 0.0
# epoch_topk_acc = 0.0
# with torch.no_grad():
#     for batch in valid_iterator:
#         predictions = model(batch.text)
#         loss = criterion(predictions, batch.label)
#         acc = categorical_accuracy(predictions, batch.label)
#         topk_acc = topk_categorical_accuracy(predictions, batch.label)
#         epoch_acc += acc.item()
#         epoch_topk_acc += topk_acc
#     print("valid acc:", epoch_acc/len(valid_iterator), "topk_acc:", epoch_topk_acc/len(valid_iterator))

# best_valid_acc = epoch_acc/len(valid_iterator)

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    print("Epoch", epoch, "Train loss", round(train_loss,5), "Acc.", round(train_acc,5),"Valid loss", round(valid_loss,5), "Acc.", round(valid_acc, 5))
    if valid_acc > best_valid_acc:
        print("Found new best model!")
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

def predict_class(sentence):
    tokenized = word_tokenize(sentence.lower())
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_preds = preds.argmax(dim=1)
    return LABEL.vocab.itos[max_preds.item()]

model.load_state_dict(torch.load(MODEL_SAVE_PATH))

while True:
    sent = input("input a sentence:")
    predicted = predict_class(sent)
    print("Predicted emoji:",predicted)

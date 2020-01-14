import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import vocab
import torchtext.data as data
import torch.nn.functional as F

import random
import math
import os
import time

# set random seeds 
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [sent len, batch size]
        embedded = self.embedding(x)
        # embedded = [sent len, batch size, emb dim]
        output, (hidden, cell) = self.lstm(embedded)
        # output = [sent len, batch size, hid dim*num directions]
        output = output.permute(1, 0, 2)
        # output = [batch size, sent len, hid dim*num directions]
        output = F.softmax(self.fc(output), dim=2)
        return output



# 支持多分类和二分类
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
 
    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
 
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
 
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
 
    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
 
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
 
        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
 
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
 
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
 
        gamma = self.gamma
 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
 
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
# ：https://blog.csdn.net/qq_33278884/article/details/91572173

def tokenizer(text):
    return text.split()

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

    batch_cnt = 0
    for batch in iterator:
        optimizer.zero_grad()

        prediction = model(batch.text)
        prediction = prediction.permute(1, 0, 2)
        prediction = prediction.reshape(-1, list(prediction.size())[2])
        print("prediction size:", prediction.size())
        print("label size:", batch.label.size())
        print("label reshaped size:", batch.label.reshape(-1).size())
        loss = criterion(prediction, batch.label.reshape(-1))
        acc = categorical_accuracy(prediction, batch.label.reshape(-1))

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        batch_cnt += 1
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
    
        for batch in iterator:
            prediction = model(batch.text)
            prediction = prediction.permute(1, 0, 2)
            prediction = prediction.reshape(-1, list(prediction.size())[2])

            loss = criterion(prediction, batch.label.reshape(-1))
            acc = categorical_accuracy(prediction, batch.label.reshape(-1))
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

TEXT = data.Field(tokenize=tokenizer, lower=True)
LABEL = data.Field(tokenize=tokenizer, lower=True)

train_data, valid_data = data.TabularDataset.splits(
    path='data', train='train.csv',
    validation='dev.csv',
    format='csv', 
    csv_reader_params={'delimiter':'\t'},
    fields=[('text',TEXT),('label',LABEL)]
)

print('train_data[0]', vars(train_data[0]))


TEXT.build_vocab(train_data, vectors='glove.42B.300d')
LABEL.build_vocab(train_data)

print(LABEL.vocab.stoi)

device = torch.device('cuda')

BATCH_SIZE = 64
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = len(LABEL.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = NER(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    device=device
)

optimizer = optim.Adam(model.parameters())
criterion = FocalLoss()
model = model.to(device)
criterion = criterion.to(device)


N_EPOCHS = 10
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'model_lstm.pt')

best_valid_loss = float('inf')
best_valid_acc = 0
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)


for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    print("Epoch", epoch, "Train loss", train_loss, "Acc.", train_acc)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    print("Epoch", epoch, "Valid loss", valid_loss, "Acc.", valid_acc)
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

        

model.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_loss, test_acc = evaluate(model, valid_iterator, criterion)
print("Test loss", test_loss, "Acc.", test_acc)

def convert2vec(sentence):
    tokenized = sentence.lower().split()
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    preds = preds.squeeze(0)
    preds = torch.argmax(preds, dim=1)
    translation = [LABEL.vocab.itos[t] for t in preds]
    return translation
    

while 1:
    sentence = input("Input a sentence:")
    translation = convert2vec(sentence)
    output = ""
    for word, tag in zip(sentence.split(), translation):
        output += word + "/" + tag + ' '
    print(output)
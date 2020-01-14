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

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiďŹed examples (p > .5), 
                                   putting more focus on hard, misclassiďŹed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1).cuda()
        else:
            if isinstance(alpha, torch.tensor):
                self.alpha = alpha
            else:
                self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = torch.tensor(class_mask).cuda()
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

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
    path='.', train='train.csv',
    validation='test.csv',
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
criterion = FocalLoss(2)
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
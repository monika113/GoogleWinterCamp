import os
import csv
import random
import re

f_train = open("train_ori.csv", "r", encoding='utf-8')
freader = csv.reader(f_train, dialect='excel')
train_header = next(freader)


out_data = []
for i, rowlist in enumerate(freader):
    out_data.append([re.sub(r"[^a-zA-Z0-9?.!,']+", " ", rowlist[-2]).lower(), rowlist[-1]])

print(out_data[0])
print(out_data[1])
print(out_data[2])

random.shuffle(out_data)

f_out = open("train.csv", 'w', newline='', encoding='utf-8')
freader = csv.writer(f_out, dialect='excel-tab')

freader.writerows(out_data[int(len(out_data)*0.8):])


f_out = open("dev.csv", 'w', newline='', encoding='utf-8')
freader = csv.writer(f_out, dialect='excel-tab')

freader.writerows(out_data[:int(len(out_data)*0.8)])



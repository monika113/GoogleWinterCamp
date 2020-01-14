#!/usr/bin/python
# -*- coding:utf8 -*-
import csv
import re 

FILE_NAME = 'train.txt'

f_in = open(FILE_NAME, 'r', encoding='utf-8')

cnt = 0 
training_data = []
word_stack = []
tag_stack = []
for line in f_in.readlines():
    splited = line.split()
    if len(splited) == 0:
        training_data.append([])
        training_data[-1].append("")
        training_data[-1].append("")
        for i, (w, t) in enumerate(zip(word_stack, tag_stack)):
            if i == 0 or i == len(word_stack) - 1:
                continue
            if not re.search(u'^[_a-zA-Z0-9\u4e00-\u9fa5]+$', w):
                continue
            training_data[-1][0] += (' ' + w.lower())
            training_data[-1][1] += (' ' + t)
        word_stack = []
        tag_stack = []
        cnt += 1
        if (cnt % 1000) == 0:
            print(cnt)
        if cnt == 200000:
            break
    elif len(splited) != 2:
        print("warning")
    else:
        word_stack.append(splited[0])
        tag_stack.append(splited[1])
    
f_out = open("train.csv", 'w', newline='', encoding='utf-8')
freader = csv.writer(f_out, dialect='excel-tab')

freader.writerows(training_data)
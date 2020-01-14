import os
import json
import time
import codecs
import re
import pickle
import random

codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)

def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  sentence = re.sub(r"([?.!,'])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  sentence = re.sub(r"[^a-zA-Z?.!,']+", " ", sentence)
  sentence = sentence.strip()
  # adding a start and an end token to the sentence
  return sentence


IN_FILE_NAME = 'personachat_self_original.json'
data = json_load(IN_FILE_NAME)

print(data.keys())
# dict_keys(['train', 'valid'])

train_data = data['valid']

questions = []
answer = []

questions = []
answers = []
cnt = 0
for item in train_data:
  history = item["utterances"][-1]["history"]
  for i in range(len(history)-1):
    ques = history[i]
    ans = history[i+1]
    if len(ques.split()) <= 40 and len(ques.split()) <= 40:
      cnt += 1
      questions.append(ques)
      answers.append(ans)
print(cnt)


out_enc = open('valid_enc.pk', 'wb')
out_dec = open('valid_dec.pk', 'wb')

pickle.dump(questions, out_enc)
pickle.dump(answers, out_dec)

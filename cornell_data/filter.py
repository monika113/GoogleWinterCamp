import re
import pickle

def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  # adding a start and an end token to the sentence
  return sentence


ENC_FILE_NAME = 'test.enc'
DEC_FILE_NAME = 'test.dec'
f_enc = open(ENC_FILE_NAME, 'r')
f_dec = open(DEC_FILE_NAME, 'r')

out_enc = open(ENC_FILE_NAME + '.pk', 'wb')
out_dec = open(DEC_FILE_NAME + '.pk', 'wb')

cnt_all = 0
cnt_valid = 0

questions = []
answers = []

for i, (que, ans) in enumerate(zip(f_enc.readlines(), f_dec.readlines())):
    if i % 2 != 0:
        continue
    cnt_all += 1
    ques = preprocess_sentence(que)
    ans = preprocess_sentence(ans)
    if len(ques.split()) <= 40 and len(ans.split()) <= 40:
        cnt_valid += 1
        questions.append(ques)
        answers.append(ans)

print(cnt_valid)
print(cnt_all)

pickle.dump(questions, out_enc)
pickle.dump(answers, out_dec)
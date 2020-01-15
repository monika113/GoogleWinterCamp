import pandas as pd
import re
import pickle


Person_name = ['Homer Simpson', 'Marge Simpson', 'Bart Simpson', 'Lisa Simpson']
person = 'Homer Simpson'
MAX_LENGTH = 40

csv_data = pd.read_csv("../data/simpsons_dataset.csv")
length, _ = csv_data.shape
print("length", length)


N = 100

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

for p in range(len(Person_name)):
    count = 0
    person = Person_name[p]
    questions = []
    answers = []
    for i in range(length):
    # for i in range(N):
        if csv_data["raw_character_text"][i] == person:
            if isinstance(csv_data["spoken_words"][i - 1], str) and csv_data["raw_character_text"][i-1] != person:
                q = preprocess_sentence(csv_data["spoken_words"][i - 1])
                a = preprocess_sentence(csv_data["spoken_words"][i])
                len_q, len_a = q.count(' '), a.count(' ')
                # cut the last sentence
                if len_q >= MAX_LENGTH:
                    q = q[q.rfind(". ")+2:]
                    len_q = q.count(' ')
                if len_a >= MAX_LENGTH:
                    a = a[a.rfind(". ")+2:]
                    len_a = a.count(' ')
                if len_q < MAX_LENGTH and len_a < MAX_LENGTH:
                    questions.append(q)
                    answers.append(a)
                    count += 1

    # save pickle
    q_out = open('../data/{}_Q'.format(person), "wb")
    pickle.dump(questions, q_out)
    a_out = open('../data/{}_A'.format(person), "wb")
    pickle.dump(answers, a_out)

    print("person",  person, "total", count, count/length)






# while i < length:
#     # start a dialog
#     if csv_data["raw_character_text"][i] == person:
#         if not start:
#             if isinstance(csv_data["spoken_words"][i-1], str):
#                 dialog.append(csv_data["spoken_words"][i-1])
#                 questions.append(csv_data["spoken_words"][i-1])
#             start = True
#         sentence = csv_data["spoken_words"][i]
#         i += 1
#         # while csv_data["raw_character_text"][i] == person:
#         #     sentence = sentence + ' ' + csv_data["spoken_words"][i]
#         #     i += 1
#         dialog.append(sentence)
#         answers.append(sentence)
#
#     # other person speaks
#     if start:
#         other_person = csv_data["raw_character_text"][i]
#         sentence = csv_data["spoken_words"][i]
#         i += 1
#         while csv_data["raw_character_text"][i] == other_person:
#             sentence = sentence + ' ' + csv_data["spoken_words"][i]
#             i += 1
#         dialog.append(sentence)
#
#         # dialog ends
#         if csv_data["raw_character_text"][i] != person:
#             start = False
#             Dialogs.append(dialog)
#             print("dialog", dialog)
#             dialog = []
#     else:
#         i += 1
#
# if start:
#     Dialogs.append(dialog)
#     print("dialog", dialog)
#
# print("Dialogs", len(Dialogs))
#
#

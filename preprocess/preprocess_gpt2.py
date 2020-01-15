import pandas as pd
import math
import nltk
from nltk.tokenize import word_tokenize

# TODO: still have the nan error
person = 'Homer Simpson'

csv_data = pd.read_csv("../data/simpsons_dataset.csv")
length, _ = csv_data.shape
print("length", length)


Dialogs = {}
Person_name = ['Homer Simpson', 'Marge Simpson', 'Bart Simpson', 'Lisa Simpson']
for person in Person_name:
    Dialogs[person] = []

start = False
i = 0
dialog = []
dialog.append('[CLS]')
while i < length:
    if i % 100 == 0:
        print ("step : ", i)
        for person in Person_name:
            print(person, ":", len(Dialogs[person]), " ; ")
    # start a dialog
    person = csv_data["raw_character_text"][i]
    if (not isinstance(person, str)) or person == "":
        dialog = []
        dialog.append('[CLS]')
        start = False
        i += 1
        continue
    sentence = csv_data["spoken_words"][i]
    sentence = sentence.lower()
    dialog = dialog + [str(x) for x in word_tokenize(sentence)] + ['[SEP]']
    if person in Person_name:
        if start:
            Dialogs[person].append(dialog)
    i += 1
    start = True

for person in Person_name:
    print(person, ":\n")
    print("length: ", len(Dialogs[person]))
    print("example: ", Dialogs[person][1])

out_path = open('../data/Dialogues', "wb")
pickle.dump(Dialogs, out_path)




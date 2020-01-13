import pandas as pd
import math

# TODO: still have the nan error
person = 'Homer Simpson'

csv_data = pd.read_csv("../data/simpsons_dataset.csv")
length, _ = csv_data.shape
print("length", length)


start = False
Dialogs = []
dialog = []
i = 0

while i < length:
    # start a dialog
    if csv_data["raw_character_text"][i] == person:
        if not start:
            if isinstance(csv_data["spoken_words"][i-1], str):
                dialog.append(csv_data["spoken_words"][i-1])
            else:
                dialog.append("XXXX")
            start = True
        sentence = csv_data["spoken_words"][i]
        i += 1
        while csv_data["raw_character_text"][i] == person:
            sentence = sentence + ' ' + csv_data["spoken_words"][i]
            i += 1
        dialog.append(sentence)

    # other person speaks
    if start:
        other_person = csv_data["raw_character_text"][i]
        sentence = csv_data["spoken_words"][i]
        i += 1
        while csv_data["raw_character_text"][i] == other_person:
            sentence = sentence + ' ' + csv_data["spoken_words"][i]
            i += 1
        dialog.append(sentence)

        # dialog ends
        if csv_data["raw_character_text"][i] != person:
            start = False
            Dialogs.append(dialog)
            print("dialog", dialog)
            dialog = []
    else:
        i += 1

if start:
    Dialogs.append(dialog)
    print("dialog", dialog)

print("Dialogs", len(Dialogs))



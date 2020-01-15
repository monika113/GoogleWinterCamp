import pickle

data_path = open("../data/Dialogues", 'rb')
Dialogs = pickle.load(data_path)

Person_name = ['Homer Simpson', 'Marge Simpson', 'Bart Simpson', 'Lisa Simpson']
for person in Person_name:
    print(person, ":")
    print("length: ", len(Dialogs[person]))
    print("example: ", Dialogs[person][1])

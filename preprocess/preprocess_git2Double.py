import pandas as pd
import math
import copy
import pickle
import random

# TODO: still have the nan error

class Data:
    def __init__(self, person):
        self.person = person
        self.candidate_pool = []
        self.cvs_data = None
        self.length = -1
        self.n_candidate = 20
        self.dataset = []
        self.Person_name = ['Homer Simpson', 'Marge Simpson', 'Bart Simpson', 'Lisa Simpson']
        self.dialog_path = '../data/Dialogues2'
        self.data_path = '../data/'+self.person.split(' ')[0]+'_gpt22'
        self.Dialogs = []
        self.person_map = {}
        for person in self.Person_name:
            self.person_map[person.lower().split(' ')[0]] = person

    def load_data(self, data_path):
        self.csv_data = pd.read_csv(data_path)
        self.length, _ = self.csv_data.shape
        print("length", self.length)

    def get_candidate_pool(self):
        i = 0
        while i < self.length:
            if self.csv_data["raw_character_text"][i] != self.person and isinstance(self.csv_data["spoken_words"][i], str):
                self.candidate_pool.append(self.csv_data["spoken_words"][i])
            i += 1
        print("candidate pool", len(self.candidate_pool))
        return self.candidate_pool

    def get_candidates(self, sentence):
        candidates = random.choices(self.candidate_pool, k=self.n_candidate - 1)
        candidates.append(sentence)
        return candidates

    def dialogue_precess(self, dialogue):
        entity = {}
        entity['personality'] = []
        entity['utterances'] = []
        if not dialogue:
            return entity
        history = []
        history.append(dialogue['utterances'][0])
        character = []
        character.append(dialogue['character'][0])
        i = 1
        person_short = self.person.lower().split(' ')[0]
        while i < len(dialogue['utterances']):
            character.append(dialogue['character'][i])
            sentence = dialogue['utterances'][i]
            if dialogue['character'][i] == person_short:
                step = {}
                step['candidates'] = self.get_candidates(sentence)
                step['history'] = copy.deepcopy(history)
                step['character'] = copy.deepcopy(character)
                entity['utterances'].append(step)
            history.append(dialogue['utterances'][i])
            i += 1
        return entity
    
    def preprocess(self):
        start = False
        i = 0
        dialogue = {}
        dialogue['utterances'] = []
        dialogue['character'] = []
        while i < self.length:
            if i % 1000 == 0:
                print ("step : ", i)
            # start a dialog
            person = self.csv_data["raw_character_text"][i]
            sentence = self.csv_data["spoken_words"][i]
            if (not isinstance(person, str)) or person == "":
                if start:
                    self.Dialogs.append(dialogue)
                dialogue = {}
                dialogue['utterances'] = []
                dialogue['character'] = []
                start = False
                i += 1
                continue
            sentence = sentence.lower()
            dialogue['utterances'].append(sentence)
            if person in self.Person_name:
                dialogue['character'].append(person.lower().split(' ')[0])
            else:
                dialogue['character'].append('other')
            i += 1
            start = True
        if start:
            self.Dialogs.append(dialogue)
        out_path = open(self.dialog_path, "wb")
        pickle.dump(self.Dialogs, out_path)
        print('Dialogs: ', len(self.Dialogs))
        print('example:', self.Dialogs[0])
        return

    def load_dialogs(self):
        data_path = open(self.dialog_path, 'rb')
        self.Dialogs = pickle.load(data_path)
        print('Dialogs: ', len(self.Dialogs))
        return
    
    def get_data(self):
        for index, dialog in enumerate(self.Dialogs):
            if index %100 == 0:
                print('step:', index)
            entity = self.dialogue_precess(dialog)
            if len(entity['utterances']) > 0:
                self.dataset.append(self.dialogue_precess(dialog))
        out_path = open(self.data_path, "wb")
        pickle.dump(self.dataset, out_path)
        print('dataset: ', len(self.dataset))
        print('example:', self.dataset[0])

if __name__ == '__main__':
    person = 'Homer Simpson'
    data = Data(person)
    data.load_data("../data/simpsons_dataset.csv")
    data.get_candidate_pool()
    data.load_dialogs()
    data.get_data()

import pickle

q_in = open("../data/Bart Simpson_Q", 'rb')
list_q = pickle.load(q_in)

print(list_q)
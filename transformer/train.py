import tensorflow as tf
import tensorflow_datasets as tfds

import os
import re
import numpy as np
import pickle
from argparse import ArgumentParser

import config
from util import get_dataset
from model import transformer


def train(inputs, outputs, pre_train=False):
    tf.keras.backend.clear_session()
    dataset, VOCAB_SIZE, _ = get_dataset(inputs, outputs)
    if pre_train:
        model = tf.keras.models.load_model(config.MODEL_PATH)
        print('load pre train model success!')
    else:
        model = transformer(
            vocab_size=VOCAB_SIZE,
            num_layers=config.NUM_LAYERS,
            units=config.UNITS,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            dropout=config.DROPOUT)
        learning_rate = model.CustomSchedule(config.D_MODEL)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        model.compile(optimizer=optimizer, loss=model.loss_function, metrics=[model.accuracy])
        print('build model success, start training...')
        model.fit(dataset, epochs=config.EPOCHS)

        model.save(config.MODEL_PATH)
        print()


if __name__ == "__main__":
    # 读入
    questions_infile = open(config.INPUT_PATH, 'rb')
    answers_infile = open(config.OUTPUT_PATH, 'rb')
    questions = pickle.load(questions_infile)
    answers = pickle.load(answers_infile)
    print('load data success!')
    train(questions, answers, pre_train=False)


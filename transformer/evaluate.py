import tensorflow as tf
import tensorflow_datasets as tfds

import os
import re
import numpy as np
import pickle

import config
from util import get_dataset, get_args
from model import transformer, CustomSchedule, loss_function, accuracy

class Chatbot:
    def __init__(self):
        self.bot = None
        self.tokenizer = None

    def evaluate(self, sentence, tokenizer, model):
        # sentence = model.preprocess_sentence(sentence)

        # tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(config.TOKENIZER_PATH)
        START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
        sentence = tf.expand_dims(
            START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

        output = tf.expand_dims(START_TOKEN, 0)

        for i in range(config.MAX_LENGTH):
            predictions = model(inputs=[sentence, output], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, END_TOKEN[0]):
                break

            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0)

    def predict(self, sentence):
        prediction = self.evaluate(sentence)

        predicted_sentence = self.tokenizer.decode(
            [i for i in prediction if i < self.tokenizer.vocab_size])

        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))

        return predicted_sentence

    def load_model(self):
        self.tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(config.EVA_TOKENIZER_PATH)
        VOCAB_SIZE = self.tokenizer.vocab_size + 2
        model = transformer(
            vocab_size=VOCAB_SIZE,
            num_layers=config.NUM_LAYERS,
            units=config.UNITS,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            dropout=config.DROPOUT)
        model.load_weights(config.EVA_MODEL_PATH)
        self.bot = model
        print('load model success')
        return self.tokenizer, self.bot


if __name__ == "__main__":
    # feed the model with its previous output
    args = get_args()
    tokenizer, model = load_model(args)

    sentence = 'I am not crazy, my mother had me tested.'
    for _ in range(5):
        sentence = predict(sentence)
        print('')

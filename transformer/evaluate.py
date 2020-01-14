import tensorflow as tf
import tensorflow_datasets as tfds

import os
import re
import numpy as np

import config
import util
import model
import pickle


def evaluate(sentence, tokenizer):
    # sentence = model.preprocess_sentence(sentence)

    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(config.TOKENIZER_PATH)
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    model = tf.keras.models.load_model(config.MODEL_PATH)

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


def predict(sentence, tokenizer):
    prediction = evaluate(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence


if __name__ == "__main__":
    # feed the model with its previous output
    sentence = 'I am not crazy, my mother had me tested.'
    for _ in range(5):
        sentence = predict(sentence)
        print('')

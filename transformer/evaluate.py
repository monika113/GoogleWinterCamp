import tensorflow as tf
import tensorflow_datasets as tfds

import torch
import torch.nn.functional as F

import os
import re
import numpy as np
import pickle

import os
cur_dir = os.path.abspath(os.path.dirname(__file__))
import sys
sys.path.append(cur_dir)
import config
from util import get_dataset, get_args
from model import transformer, CustomSchedule, loss_function, accuracy


class Chatbot:
    def __init__(self):
        self.bot = None
        self.tokenizer = None

    def evaluate(self, sentence, top_k=8, top_p=0, threshold=-float('Inf'), filter_value=-float('Inf')):
        # sentence = model.preprocess_sentence(sentence)

        # tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(config.TOKENIZER_PATH)
        START_TOKEN, END_TOKEN = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]
        sentence = tf.expand_dims(
            START_TOKEN + self.tokenizer.encode(sentence) + END_TOKEN, axis=0)

        output = tf.expand_dims(START_TOKEN, 0)

        # for i in range(config.MAX_LENGTH):
        #     predictions = self.bot(inputs=[sentence, output], training=False)

        #     # select the last word from the seq_len dimension
        #     predictions = predictions[:, -1:, :]
        #     predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        #     # return the result if the predicted_id is equal to the end token
        #     if tf.equal(predicted_id, END_TOKEN[0]):
        #         break

        #     # concatenated the predicted_id to the output which is given to the decoder
        #     # as its input.
        #     output = tf.concat([output, predicted_id], axis=-1)
        for i in range(config.MAX_LENGTH):
            predictions = self.bot(inputs=[sentence, output], training=False)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]


            # logits = torch.tensor(tf.make_ndarray(predictions.op.get_attr('value')))
            logits = torch.tensor(np.array(predictions)).squeeze(0).squeeze(0)
            # print("logits size:", logits.size())

            if top_k > 0:
                # Remove all tokens with a probability less than the last token in the top-k tokens
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = filter_value
            
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # print("cumulative prob size:", cumulative_probabilities.size())

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probabilities > top_k
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Back to unsorted indices and set them to -infinity
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                # print("indices to remove:", indices_to_remove.size())
                logits[indices_to_remove] = filter_value
            
            indices_to_remove = logits < threshold
            logits[indices_to_remove] = filter_value

            probabilities = F.softmax(logits, dim=-1)
            predicted_id = torch.multinomial(probabilities, 1).item()


            # # use top-p 
            # sorted_logits = tf.sort(predictions, direction='DESCENDING')
            # sorted_indices = tf.argsort(predictions, direction='DESCENDING')
            # cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1, exclusive=True)
            # sorted_indices_to_remove = cumulative_probs > 0.9
            # indices_to_remove = sorted_indices[sorted_indices_to_remove]
            # indices_to_remove = tf.make_ndarray(indices_to_remove)
            # predictions[indices_to_remove] = -float('Inf')
            # probabilities = tf.nn.softmax(predictions, axis=-1)
            # sampled = tf.random.categorical(probabilities, 1)
            # predicted_id = tf.cast(tf.argmax(sampled, axis=-1), tf.int32)


            # return the result if the predicted_id is equal to the end token
            # print("type end_token[0]", type(END_TOKEN[0]))
            if tf.equal(predicted_id, END_TOKEN[0]):
                break

            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            # print("output size:", tf.shape(output))
            output = tf.concat([output, np.array([[predicted_id]])], axis=-1)

        return tf.squeeze(output, axis=0)


    def predict(self, sentence):
        prediction = self.evaluate(sentence)

        predicted_sentence = self.tokenizer.decode(
            [i for i in prediction if i < self.tokenizer.vocab_size])

        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))

        return predicted_sentence

    def load_model(self, args = None):
        if args:
            load_tokenizer_path = args.load_tokenizer_path
            pre_train_model_path = args.pre_train_model_path
        else:
            load_tokenizer_path = config.TOKENIZER_PATH
            pre_train_model_path = config.MODEL_PATH
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        load_tokenizer_path_new = cur_dir + '/' + load_tokenizer_path
        self.tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(load_tokenizer_path_new)
        VOCAB_SIZE = self.tokenizer.vocab_size + 2
        model = transformer(
            vocab_size=VOCAB_SIZE,
            num_layers=config.NUM_LAYERS,
            units=config.UNITS,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            dropout=config.DROPOUT)
        pre_train_model_path_new = cur_dir + '/' +pre_train_model_path
        model.load_weights(pre_train_model_path_new)
        self.bot = model
        print('load model success')
        return self.tokenizer, self.bot


if __name__ == "__main__":
    # feed the model with its previous output
    args = get_args()
    bot = Chatbot()
    bot.load_model(args)
    while True:
        sentence = input('say something: ')
        topk = input('topk')
        topp = input('topp')
        prediction = self.evaluate(sentence, top_k=int(topk), top_p=int(topp))

        predicted_sentence = self.tokenizer.decode(
            [i for i in prediction if i < self.tokenizer.vocab_size])

        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))
        print('')

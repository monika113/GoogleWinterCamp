import config
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle


def build_tokenizer(inputs, outputs):
    # Build tokenizer using tfds for both questions and answers
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        inputs + outputs, target_vocab_size=2**13)

    tokenizer.save_to_file(config.TOKENIZER_PATH)

    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2

    return tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE


def tokenize_and_filter(inputs, outputs, tokenizer, START_TOKEN, END_TOKEN):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        # check tokenized sentence max length
        if len(sentence1) <= config.MAX_LENGTH and len(sentence2) <= config.MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=config.MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=config.MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs


def get_dataset(inputs, outputs):
    tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE = build_tokenizer(inputs, outputs)
    tokenized_inputs, tokenized_outputs = tokenize_and_filter(inputs, outputs, tokenizer, START_TOKEN, END_TOKEN)
    print('Vocab size: {}'.format(VOCAB_SIZE))
    print('Number of samples: {}'.format(len(inputs)))
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': inputs,
            'dec_inputs': outputs[:, :-1]
        },
        {
            'outputs': outputs[:, 1:]
        },
    ))

    dataset = dataset.cache()
    dataset = dataset.shuffle(config.BUFFER_SIZE)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, VOCAB_SIZE, tokenizer


MAX_LENGTH = 40

BATCH_SIZE = 64
BUFFER_SIZE = 20000

NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

EPOCHS = 20

MODEL_PATH = 'model/step01.h5'
TOKENIZER_PATH = 'tokenizer/token01'
INPUT_PATH = '../cornell_data/train.enc.pk'
OUTPUT_PATH = '../cornell_data/train.dec.pk'

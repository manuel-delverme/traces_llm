TEXT_PADDING_ID = Exception("Use tokenizer pad id")
GPT2_VOCAB_SIZE = 50257
VOCAB_SIZE = GPT2_VOCAB_SIZE  # sys.maxsize  # 26 ** 3
assert VOCAB_SIZE > 1
DATASET_SIZE = None  # 32777

# Max number of characters in a token, a word like "incorporated" would net 12 images and traces
MAX_CHARS_PER_TOKEN = 16
EMPTY_CHAR = " "
IMG_PATH = 'data/images'
TRACES_PATH = 'data/traces'
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
TEXT_DATASET_PATH = 'tiny_shakespeare.txt'

optimized_metric = "val_loss"
optimization_mode = "min"
eager_rate = 0.50

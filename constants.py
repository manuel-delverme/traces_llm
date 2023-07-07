TEXT_PADDING_ID = Exception("Use tokenizer pad id")
GPT2_VOCAB_SIZE = 50257
VOCAB_SIZE = GPT2_VOCAB_SIZE  # sys.maxsize  # 26 ** 3
assert VOCAB_SIZE > 1
DATASET_SIZE = None  # 32777

# Max number of characters in a token, a word like "incorporated" would net 12 images and traces
MAX_CHARS_PER_TOKEN = 16
EMPTY_CHAR = " "
IMG_PATH = '/home/delverme/Downloads/images_background_small1'
TRACES_PATH = '/home/delverme/Downloads/strokes_background_small1/strokes_background_small1'
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
TEXT_DATASET_PATH = 'tiny_shakespeare.txt'

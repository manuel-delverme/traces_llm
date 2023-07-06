TEXT_PADDING_ID = 0  # TODO: Change this to the correct padding ID for your tokenizer
GPT2_VOCAB_SIZE = 50257
VOCAB_SIZE = 26 ** 3
assert VOCAB_SIZE > 1
DATASET_SIZE = None  # 32777

# Max number of characters in a token, a word like "incorporated" would net 12 images and traces
MAX_CHARS_PER_TOKEN = 12

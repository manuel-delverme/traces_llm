from transformers import GPT2Tokenizer

tokenizer = None


def get_default_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

from collections import deque

import numpy as np
import torch
from transformers import BatchEncoding

import constants
import dataset
import hyper
import presets
from dataset import DataSample
from text_only_baseline import GPT2FineTuning


class HandwritingRecognizer:
    def __init__(self, model: GPT2FineTuning):
        self.tokenizer = presets.get_default_tokenizer()

        self.char_history = None
        self.token_history = None

        self.model = model
        self.model.eval()
        # TODO: use the tokenizer from the model
        self.data_spec = model.data_spec

        self.reset()

    def update_motor_history_and_preprocess(self, token_strokes):
        assert len(token_strokes.shape) == 4, "num_chars x num_strokes=1 x num_points x (x, y, t)"
        # assert batched_char_trace.shape[0] == 1, "During evaluation we assume one trace is a character"

        for stroke_for_char in token_strokes:
            char_trace = dataset.process_strokes(stroke_for_char)
            char_trace = torch.tensor(char_trace, dtype=torch.float32).unsqueeze(0)
            self.char_history.append(char_trace)

        char_history = torch.concat(list(self.char_history))
        motor_trace = dataset.pad_motor_trace(char_history, eager_rate=1.)
        return motor_trace.unsqueeze(0)

    def resample_stroke(self, motor_context):
        return dataset.resample_stroke(motor_context, self.data_spec.points_in_motor_sequence)

    @torch.no_grad()
    def update_history_and_predict(self, token_strokes):
        txt_context = torch.tensor(list(self.token_history)).unsqueeze(0)
        token_context = BatchEncoding({
            "input_ids": txt_context,
            "attention_mask": torch.ones((1, hyper.TOKEN_CONTEXT_LEN), dtype=torch.long)
        })
        # TODO: where is the token_history updated?

        motor_context = self.update_motor_history_and_preprocess(token_strokes)
        postprocessed_sample = DataSample(
            token_context=token_context,
            motor_context=motor_context,
        )
        prediction = self.model(postprocessed_sample)
        token_idx = prediction.argmax(dim=1).item()
        p_token = prediction.softmax(axis=1)[:, token_idx]
        token = self.tokenizer.decode(token_idx)
        return token, p_token

    def next_token(self):
        self.char_history.clear()

    def reset(self):
        pad_token = self.tokenizer.pad_token_id
        self.char_history = deque(maxlen=constants.MAX_CHARS_PER_TOKEN)
        self.token_history = deque(maxlen=hyper.TOKEN_CONTEXT_LEN, iterable=[pad_token] * hyper.TOKEN_CONTEXT_LEN)

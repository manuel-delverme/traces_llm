from collections import deque

import torch
from transformers import BatchEncoding

import dataset
import hyper
import presets
from dataset import DataSample
from models.trunk import GPT2FineTuning


class HandwritingRecognizer:
    def __init__(self, model: GPT2FineTuning):
        self.tokenizer = presets.get_default_tokenizer()

        self.token_history = None

        self.model = model
        self.model.eval()
        # TODO: use the tokenizer from the model
        self.data_spec = model.data_spec

        self.reset()

    @staticmethod
    def preprocess_motor_history(token_strokes):
        assert len(token_strokes.shape) == 4, "num_chars x num_strokes=1 x num_points x (x, y, t)"
        # assert batched_char_trace.shape[0] == 1, "During evaluation we assume one trace is a character"
        char_traces_history = []

        for stroke_for_char in token_strokes:
            char_trace = dataset.process_strokes(stroke_for_char)
            char_trace = torch.tensor(char_trace, dtype=torch.float32).unsqueeze(0)
            char_traces_history.append(char_trace)

        char_history = torch.concat(list(char_traces_history))
        motor_trace = dataset.pad_motor_trace(char_history, eager_rate=1.)
        return motor_trace.unsqueeze(0)

    def resample_stroke(self, motor_context):
        return dataset.resample_stroke(motor_context, self.data_spec.points_in_motor_sequence)

    @torch.no_grad()
    def update_history_and_predict(self, token_strokes, original_trace):
        txt_context = torch.tensor(list(self.token_history)).unsqueeze(0)
        print(f"text token context: {self.tokenizer.decode(txt_context.squeeze(0)).replace('<|endoftext|>', '_')}")
        token_context = BatchEncoding({
            "input_ids": txt_context,
            "attention_mask": torch.ones((1, hyper.TOKEN_CONTEXT_LEN), dtype=torch.long)
        })
        motor_context = self.preprocess_motor_history(token_strokes)

        postprocessed_sample = DataSample(
            token_context=token_context,
            motor_context=motor_context,
        )
        prediction = self.model(postprocessed_sample)
        top_k = prediction.topk(k=10, dim=1).indices[0].tolist()
        filtered_tokens = self.tokenizer.convert_ids_to_tokens(top_k)

        # TODO: figure out top 10 tokens and print them
        p_tokens = prediction.softmax(axis=1)[0][top_k].tolist()
        return filtered_tokens, p_tokens

    def reset(self):
        self.token_history = deque(
            maxlen=hyper.TOKEN_CONTEXT_LEN, iterable=[self.tokenizer.pad_token_id] * hyper.TOKEN_CONTEXT_LEN)

import random
import time
from collections import deque

import numpy as np
import torch
from transformers import BatchEncoding

import constants
import dataset
import hyper
from dataset import DataSpec, DataSample
from text_only_baseline import GPT2FineTuning


class MockModel:
    def __init__(self, data_spec: DataSpec):
        self.data_spec = data_spec
        pass

    def eval(self):
        pass

    def __call__(self, mouse_positions):
        # For a mock model, we'll just return a random number
        return torch.rand(1)


class HandwritingRecognizer:
    def __init__(self, model: GPT2FineTuning):
        self.context_window = deque(maxlen=constants.MAX_CHARS_PER_TOKEN)
        self.model = model
        self.data_spec = model.data_spec

    def preprocess_mouse_trace(self, mouse_positions):
        # During evaluation we assume one stroke is one trace.
        stroke = self.resample_stroke(mouse_positions)
        stroke = torch.tensor(stroke, dtype=torch.float32).unsqueeze(0)

        self.context_window.append(stroke)

        context = torch.concat(list(self.context_window))
        motor_trace = dataset.pad_motor_trace(context, eager_rate=1.)
        return motor_trace

    def resample_stroke(self, motor_context):
        motor_context = dataset.resample_stroke(motor_context, self.data_spec.points_in_motor_sequence)
        return motor_context

    @torch.no_grad()
    def predict(self, mouse_positions):
        past_tokens = torch.randint(constants.VOCAB_SIZE, (1, hyper.TOKEN_CONTEXT_LEN))
        token_context = BatchEncoding({
            "input_ids": past_tokens,
            "attention_mask": torch.ones((1, hyper.TOKEN_CONTEXT_LEN), dtype=torch.long)
        })

        motor_context = self.preprocess_mouse_trace(mouse_positions).unsqueeze(0)
        postprocessed_sample = DataSample(
            token_context=token_context,
            motor_context=motor_context,
        )
        prediction = self.model(postprocessed_sample)
        return prediction


class MockGUI:
    def __init__(self, data_spec: DataSpec):
        self.mouse_positions = []
        # model = MockModel(data_spec)
        model = GPT2FineTuning(data_spec)
        model.eval()
        self.recognizer = HandwritingRecognizer(model)

    def move_mouse(self, x, y, t):
        self.mouse_positions.append((x, y, t))

    def recognize_handwriting(self):
        prediction = self.recognizer.predict(np.array(self.mouse_positions))
        # Clear the mouse positions for the next recognition
        self.mouse_positions = []
        return prediction


def main():
    data_spec = dataset.DataSpec(
        use_images=False,
        use_motor_traces=True,
    )
    gui = MockGUI(data_spec=data_spec)
    t = time.time()

    # Simulate moving the mouse
    for i in range(100):
        t += random.random() / 100
        gui.move_mouse(i, i, t)

    prediction = gui.recognize_handwriting()
    print(prediction)


if __name__ == "__main__":
    main()

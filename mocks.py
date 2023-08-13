import random
import time

import numpy as np
import torch

from dataset import DataSpec
from gui import BaseGUI
from preprocess import HandwritingRecognizer
from text_only_baseline import GPT2FineTuning
from user_model import UserInteraction


class MockModel:
    def __init__(self, data_spec: DataSpec):
        self.data_spec = data_spec
        pass

    def eval(self):
        pass

    def __call__(self, mouse_positions):
        # For a mock model, we'll just return a random number
        return torch.rand(1)


class MockGUI(BaseGUI):
    def __init__(self, model: GPT2FineTuning, user_interaction: UserInteraction):
        super().__init__()
        self.recognizer = HandwritingRecognizer(model)
        self.user_model = user_interaction
        self.width = 100
        self.height = 100

    def recognize_handwriting(self):
        token_motor_trace = np.array(self.char_strokes)
        prediction = self.recognizer.update_history_and_predict(token_motor_trace)
        # Clear the mouse positions for the next recognition
        self.mouse_positions = []
        return prediction

    def run_once(self):
        t = time.time()
        for i in range(100):
            t += random.random() / 100
            self.track_move_mouse(i, i, t)
            if i % 10 == 0:
                self.user_model.reset(self.width, self.height)

        prediction = self.recognize_handwriting()
        return prediction

    def run(self):
        self.user_model.reset(self.width, self.height)
        for i in range(10):
            self.run_once()

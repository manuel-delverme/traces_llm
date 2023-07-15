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
    def __init__(self, model: GPT2FineTuning):
        self.mouse_positions = []
        model.eval()
        self.recognizer = HandwritingRecognizer(model)

    def move_mouse(self, x, y, t):
        self.mouse_positions.append((x, y, t))

    def recognize_handwriting(self):
        prediction = self.recognizer.predict(np.array(self.mouse_positions))
        # Clear the mouse positions for the next recognition
        self.mouse_positions = []
        return prediction

    def run_once(self):
        t = time.time()
        for i in range(100):
            t += random.random() / 100
            self.move_mouse(i, i, t)

        prediction = self.recognize_handwriting()
        return prediction

    def run(self):
        for i in range(10):
            self.run_once()


class PygameGUI:
    def __init__(self, model):
        import pygame
        pygame.init()
        self.model = model
        self.model.eval()
        self.WIDTH, self.HEIGHT = 600, 600
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.mouse_positions = []
        self.recognizer = HandwritingRecognizer(self.model)

    def move_mouse(self, x, y, t):
        self.mouse_positions.append((x, y, t))

    def recognize_handwriting(self):
        prediction = self.recognizer.predict(np.array(self.mouse_positions))
        self.mouse_positions = []
        return prediction

    def run_once(self):
        import pygame
        should_continue = True
        pygame.event.clear()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_continue = False
            if event.type == pygame.MOUSEMOTION:
                x, y = pygame.mouse.get_pos()
                t = pygame.time.get_ticks()  # Get the current time
                pygame.draw.circle(self.surface, self.WHITE, (x, y), 10)
                self.move_mouse(x, y, t)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                prediction = self.recognize_handwriting()
                # TODO: process the prediction and display it
                print(prediction)

        self.window.fill(self.WHITE)
        self.window.blit(self.surface, (0, 0))
        pygame.display.flip()
        return should_continue

    def run(self):
        import pygame
        clock = pygame.time.Clock()
        while self.run_once():
            clock.tick(60)
        pygame.quit()


# Usage:
# model = GPT2FineTuning(DataSpec(
#    use_images=False,
#    use_motor_traces=True,
# ))
# gui = PygameGUI(model)
# gui.run()


def main():
    data_spec = dataset.DataSpec(
        use_images=False,
        use_motor_traces=True,
    )
    model = GPT2FineTuning(data_spec)
    # gui = MockGUI(model)
    gui = PygameGUI(model)
    prediction = gui.run_once()

    print(prediction)


if __name__ == "__main__":
    main()

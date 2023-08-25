import random
import time

import numpy as np
import pygame

from dataset import revert_preprocess_trace


class UserInteraction:
    def get_pos(self):
        raise NotImplementedError

    def reset(self, width, height):
        raise NotImplementedError


class RealUserInteraction(UserInteraction):
    def get_pos(self):
        return pygame.mouse.get_pos()


class OfflineUserInteraction(UserInteraction):
    def __init__(self, dataset):
        self.dataset = dataset
        self.token_traces = None
        self.next_token = None
        self.current_step = None
        self.token_events = None
        self.postprocessed_token_trace = None

    def reset(self, width, height):
        padded_token_traces, next_token = self.get_new_trace()
        self.postprocessed_token_trace = padded_token_traces

        next_token = self.dataset.tokenizer.decode(next_token)
        char_traces = [trace for trace in padded_token_traces if not (trace == 0).all()]
        if not char_traces:
            return self.reset(width, height)

        mouse_stokes = []

        padding = np.array([20., 0])
        leftmost_point = padding.copy()
        leftmost_point[1] = 20.

        for char_trace in char_traces:
            scaled_trace = revert_preprocess_trace(char_trace)

            # Next character starts at the end of the current one
            max_extent = scaled_trace.max(axis=0)
            max_extent[1] = 0  # we don't move the pen up and down

            mouse_char_trace = []

            for mouse_pos in scaled_trace:
                adjusted_pos = mouse_pos + leftmost_point
                if min(adjusted_pos) < 0:
                    print(adjusted_pos.numpy())
                mouse_char_trace.append(adjusted_pos)

            mouse_char_trace = np.array(mouse_char_trace)
            if mouse_char_trace[:, 0].max() > width:
                mouse_char_trace[:, 0] *= width / mouse_char_trace[:, 0].max()
            if mouse_char_trace[:, 1].max() > height:
                mouse_char_trace[:, 1] *= height / mouse_char_trace[:, 1].max()

            leftmost_point += max_extent + padding
            mouse_stokes.append(mouse_char_trace)

        # TODO: this is ok for GUI but not for decoding

        self.token_events = []
        for mouse_stroke in mouse_stokes:
            self.token_events.append(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1}))
            for pos in mouse_stroke:
                self.token_events.append(pygame.event.Event(pygame.MOUSEMOTION, {"pos": pos}))
            self.token_events.append(pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1}))
        self.token_events.append(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_SPACE}))
        self.next_token = next_token

    def get_new_trace(self):
        index = random.randint(0, len(self.dataset) - 1)
        # sample = DataSample(**self.dataset[index])
        # motor_traces, labels = sample.motor_context, sample.labels
        sample = self.dataset[index]
        motor_traces, labels = sample["motor_context"], sample["labels"]
        return motor_traces[0].numpy(), int(labels[0])

    def get_event(self):
        if not self.token_events:
            return None
        time.sleep(1 / 120)
        event = self.token_events.pop(0)
        return event

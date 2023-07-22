import random

import numpy as np
import pygame
import torch

from dataset import DataSample


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
        self.current_trace = None
        self.next_token = None
        self.current_step = None
        # self.reset()

    def reset(self, width, height):
        padded_token_traces, next_token = self.get_new_trace()
        next_token = self.dataset.tokenizer.decode(next_token)
        token_traces = [trace for trace in padded_token_traces if not (trace == 0).all()]
        if not token_traces:
            return self.reset(width, height)

        mouse_positions = []

        padding = torch.tensor([20., 0])
        char_offset = padding.clone()
        char_offset[1] = 20.

        for trace in token_traces:
            trace -= torch.min(trace, dim=0).values
            max_extent = torch.max(trace, dim=0).values
            max_extent[1] = 0

            for mouse_pos in trace:
                adjusted_pos = mouse_pos + char_offset
                mouse_positions.append(adjusted_pos.numpy())

            char_offset += max_extent + padding

        # rot90 2 times
        # mouse_positions = np.array([(y, x) for x, y in mouse_positions])

        # fliplr
        # width = mouse_positions[:, 0].max()
        # mouse_positions[:, 0] = width - mouse_positions[:, 0]

        # Rescale mouse_positions to fit the screen (width, height)
        mouse_positions = np.array(mouse_positions)
        if mouse_positions[:, 0].max() > width:
            mouse_positions[:, 0] *= width / mouse_positions[:, 0].max()
        if mouse_positions[:, 1].max() > height:
            mouse_positions[:, 1] *= height / mouse_positions[:, 1].max()
        # TODO: this is ok for GUI but not for decoding

        self.current_trace = mouse_positions.tolist()
        self.next_token = next_token

    def get_new_trace(self):
        index = random.randint(0, len(self.dataset) - 1)
        sample = DataSample(**self.dataset[index])
        motor_traces, labels = sample.motor_context, sample.labels
        return motor_traces[0], labels[0]

    def get_pos(self):
        if not self.current_trace:
            return None
        pos = self.current_trace.pop(0)
        return pos

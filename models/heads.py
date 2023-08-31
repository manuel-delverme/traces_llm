import torch

import constants
import dataset
from constants import VOCAB_SIZE


class Permute(torch.nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def get_text_head(input_size, hidden_size, num_layers):
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(input_size, hidden_size),

        torch.nn.ReLU(),
        *[torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        ) for _ in range(num_layers)],

        torch.nn.Linear(hidden_size, VOCAB_SIZE, bias=False),
    )


def get_motor_tower(data_spec, hidden_size, num_layers, features_size):
    point_dimensions = 2
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(
            point_dimensions * data_spec.points_in_motor_sequence * constants.MAX_CHARS_PER_TOKEN,
            hidden_size),
        torch.nn.ReLU(),
        *[torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        ) for _ in range(num_layers)],
        torch.nn.Linear(hidden_size, features_size),
    )


def get_image_tower(data_spec: dataset.DataSpec, hidden_size, num_layers, features_size):
    layers = [
        Permute(0, 2, 1, 3, 4),  # Swap sequence and channel dimensions
        torch.nn.Conv3d(data_spec.image_channels, hidden_size, kernel_size=(3, 3, 3), stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
    ]

    for _ in range(num_layers):
        layers.extend([
            torch.nn.Conv3d(hidden_size, hidden_size, kernel_size=(3, 3, 3), stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        ])

    sequence_length = constants.MAX_CHARS_PER_TOKEN

    # Compute the size of the height and width after all max pooling operations
    side_after_pooling = data_spec.image_side // (2 ** (num_layers + 1))

    feature_map_size = hidden_size * sequence_length * side_after_pooling ** 2

    layers.extend([
        torch.nn.Flatten(),
        torch.nn.Linear(feature_map_size, features_size)
    ])

    return torch.nn.Sequential(*layers)

import collections

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

from utils import DataSample


def visualize_image(images):
    img_grid = torchvision.utils.make_grid(images, nrow=1, normalize=False, pad_value=1, value_range=(-1, 1))
    img_width = img_grid.shape[2]
    img_height = img_grid.shape[1]
    fig = plt.figure(figsize=(img_width / img_height, 1))
    ax = plt.Axes(fig, [0., 0., 1., 1.], )
    ax.set_axis_off()
    fig.add_axes(ax)
    img_grid = (img_grid - img_grid.min())
    img_grid = img_grid / img_grid.max()
    ax.imshow(img_grid.permute(1, 2, 0), interpolation='none')
    return images


def visualize_motor_context(ax, motor_context):
    context_sequence_len, motor_sequence_length, _ = motor_context.shape

    time_step_indices = np.arange(motor_sequence_length)
    x_cumulative = collections.deque([0, ], maxlen=3)
    y_cumulative = collections.deque([0, ], maxlen=3)

    matplotlib_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


    offset = 0
    for context_idx in range(context_sequence_len):
        # Loop through the selected time steps
        color = matplotlib_color_cycle[context_idx % len(matplotlib_color_cycle)]

        token_trace = motor_context[context_idx]
        next_offset = 0

        for seq_index in time_step_indices[1:]:
            # Extract the x and y coordinates at the current time step
            x, y = token_trace[seq_index]
            x_cumulative.append(x.item())
            y_cumulative.append(y.item())

            # Calculate differences for arrow
            dx = x_cumulative[-1] - x_cumulative[-2]
            dy = y_cumulative[-1] - y_cumulative[-2]
            ax.quiver(
                x_cumulative[-2] + offset,
                y_cumulative[-2],
                dx, dy, angles='xy', scale_units='xy', scale=1, color=color)
            next_offset = max(next_offset, x_cumulative[-1])
        offset += next_offset




def visualize_one_sample(input_data_sample: DataSample, tokenizer, num_samples=4):
    token_dict = {v: k for k, v in tokenizer.get_vocab().items()}
    text_context_ids: torch.Tensor = input_data_sample.text_context_ids
    labels = input_data_sample.labels

    # Create a figure
    fig = plt.figure(figsize=(20, 5 * num_samples))

    for sample_idx in range(num_samples):
        # Set title as token history
        tokens = text_context_ids[sample_idx].tolist()
        token_history = [token_dict[token] if token_dict else str(token) for token in tokens]
        token_str = " ".join(token_history)
        label_str = token_dict[labels[sample_idx].item()]

        # Replace ! with a symbol, replace non alphanumeric characters with empty string
        def filter_chars(c):
            if c == '!':
                return '␣'
            elif c.isalnum() and c.isascii():
                return c
            elif c == ' ':
                return ' '
            else:
                return ''

        token_str = "".join([filter_chars(c) for c in token_str])
        label_str = "".join([filter_chars(c) for c in label_str])

        # Plot image
        ax_image = fig.add_subplot(num_samples, 2, 2 * sample_idx + 1)
        images = input_data_sample.images[sample_idx]
        nrow = int(np.sqrt(images.shape[0]))
        img_grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=False, pad_value=0, value_range=(-1, 1))
        img_grid -= img_grid.min()
        img_grid /= img_grid.max()
        # Flip black and white
        img_grid = 1 - img_grid
        ax_image.imshow(img_grid.permute(1, 2, 0), interpolation='none')
        ax_image.set_title(token_str + f" => {label_str}", fontsize=12)
        ax_image.axis('off')

        ax_motor = fig.add_subplot(num_samples, 2, 2 * sample_idx + 2)
        motor_context = input_data_sample.motor_context[sample_idx]
        visualize_motor_context(ax_motor, motor_context)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    # Show the plot
    plt.show()

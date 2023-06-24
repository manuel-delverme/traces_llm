import dataclasses
from sys import platform as sys_pf

import matplotlib
import numpy as np
import torch

if sys_pf == 'darwin':
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


@dataclasses.dataclass
class DataSample:
    images: torch.Tensor
    motor_context: torch.Tensor
    text_context_ids: torch.Tensor
    labels: torch.Tensor


def embeddings_to_tokens(embeddings):
    # Round the embeddings to the nearest integer to produce tokens
    # (This is a placeholder and should be replaced with your actual method)
    embedding_tokens = torch.round(embeddings)
    return embedding_tokens


def concatenate_tokens(input_tokens, embedding_tokens):
    # Concatenate the input tokens and the embedding tokens
    # (This is a placeholder and should be replaced with your actual method)
    # input_tokens [3x1,128], embedding_tokens [3,64]
    concatenated_tokens = torch.cat((input_tokens, embedding_tokens.unsqueeze(1)), dim=2)
    return concatenated_tokens


def process_batch(batch: DataSample, device):
    # A batch is [batch, time, ...]

    images = batch.images.to(device)
    text_input = batch.text_context_ids
    labels = batch.labels
    motor_context = batch.motor_context

    # Turn the timestep into another batch dimension

    # [batch, time, ...]  -> [batch * time, ...]

    images = images.view(-1, *images.shape[2:])
    text_input = text_input.view(-1, *text_input.shape[2:])
    motor_context = motor_context.view(-1, *motor_context.shape[2:])
    labels = labels.view(-1)

    new_batch = DataSample(images=images, motor_context=motor_context, text_context_ids=text_input, labels=labels)

    return new_batch, labels


# ---
# Demo for how to load image and stroke data for a character
# ---

# Plot the motor trajectory over an image
#
# Input
#  I [105 x 105 nump] grayscale image
#  drawings: [ns list] of strokes (numpy arrays) in motor space
#  lw : line width
def plot_motor_to_image(I, drawing, lw=2):
    drawing = [d[:, 0:2] for d in drawing]  # strip off the timing data (third column)
    drawing = [space_motor_to_img(d) for d in drawing]  # convert to image space
    plt.imshow(I, cmap='gray')
    ns = len(drawing)
    for sid in range(ns):  # for each stroke
        plot_traj(drawing[sid], get_color(sid), lw)
    plt.xticks([])
    plt.yticks([])


# Plot individual stroke
#
# Input
#  stk: [n x 2] individual stroke
#  color: stroke color
#  lw: line width
def plot_traj(stk, color, lw):
    n = stk.shape[0]
    if n > 1:
        plt.plot(stk[:, 0], stk[:, 1], color=color, linewidth=lw)
    else:
        plt.plot(stk[0, 0], stk[0, 1], color=color, linewidth=lw, marker='.')


# Color map for the stroke of index k
def get_color(k):
    scol = ['r', 'g', 'b', 'm', 'c']
    ncol = len(scol)
    if k < ncol:
        out = scol[k]
    else:
        out = scol[-1]
    return out


# convert to str and add leading zero to single digit numbers
def num2str(idx):
    if idx < 10:
        return '0' + str(idx)
    return str(idx)


# Load binary image for a character
#
# fn : filename
def load_img(fn):
    I = plt.imread(fn)
    I = np.array(I, dtype=bool)
    return I


# Load stroke data for a character from text file
#
# Input
#   fn : filename
#
# Output
#   motor : list of strokes (each is a [n x 3] numpy array)
#      first two columns are coordinates
#	   the last column is the timing data (in milliseconds)
def load_motor(fn):
    motor = []
    with open(fn, 'r') as fid:
        lines = fid.readlines()
    lines = [l.strip() for l in lines]
    for myline in lines:
        if myline == 'START':  # beginning of character
            stk = []
        elif myline == 'BREAK':  # break between strokes
            stk = np.array(stk)
            motor.append(stk)  # add to list of strokes
            stk = []
        else:
            arr = np.fromstring(myline, dtype=float, sep=',')
            stk.append(arr)
    return motor


#
# Map from motor space to image space (or vice versa)
#
# Input
#   pt: [n x 2] points (rows) in motor coordinates
#
# Output
#  new_pt: [n x 2] points (rows) in image coordinates
def space_motor_to_img(pt):
    pt[:, 1] = -pt[:, 1]
    return pt


def space_img_to_motor(pt):
    pt[:, 1] = -pt[:, 1]
    return

import dataclasses
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

import constants
from utils import DataSample

from torchvision import transforms
from torch import Tensor


class MultimodalTransform:
    def __init__(self, image_transform, trace_transform):
        self.image_transform = image_transform
        self.trace_transform = trace_transform

    def __call__(self, images, motor_traces):
        image = self.image_transform(images)
        trace = self.trace_transform(motor_traces)
        return image, trace


# TODO: maybe resample strokes to be uniform in space?
def resample_stroke(stroke, num_samples=100):
    """
    Resamples a stroke to be uniform in time.

    stroke: numpy array of shape (n, 3) where n is the number of points.
            Each row is (x, y, timestamp).
    num_samples: number of samples to interpolate the stroke to.

    returns: resampled_stroke, numpy array of shape (num_samples, 2)
             where each row is (x, y).
    """

    # Extract x, y coordinates and timestamps
    x, y, t = stroke.T

    # Create a new timeline that's uniform
    new_t = np.linspace(t[0], t[-1], num_samples)

    # Interpolate x and y coordinates at the new timeline
    new_x = np.interp(new_t, t, x)
    new_y = np.interp(new_t, t, y)

    # Stack x and y to create the resampled stroke
    resampled_stroke = np.stack([new_x, new_y], axis=1)

    # plt.plot(stroke[:, 0], stroke[:, 1], 'bo')
    # plt.plot(resampled_stroke[:, 0], resampled_stroke[:, 1], 'ro')
    # plt.show()

    return resampled_stroke


# Function to normalize a single trace
def normalize_trace(trace, min_x, max_x, min_y, max_y):
    """
    Normalize the x, y coordinates of a single trace to be between 0 and 1.

    trace: numpy array of shape (n, 2) where n is the number of points.
           Each row is (x, y).

    min_x, max_x, min_y, max_y: The minimum and maximum values of x and y respectively,
                                to be used for normalization.

    returns: normalized_trace, numpy array of shape (n, 2) where each row is (normalized_x, normalized_y).
    """
    normalized_x = (trace[:, 0] - min_x) / (max_x - min_x)
    normalized_y = (trace[:, 1] - min_y) / (max_y - min_y)

    normalized_trace = np.stack([normalized_x, normalized_y], axis=1)

    return normalized_trace


class OmniglotDataset:
    def __init__(self, img_dir, stroke_dir, transforms: MultimodalTransform, alphabet_name="Latin"):
        self.img_dir = os.path.join(img_dir, alphabet_name)
        self.stroke_dir = os.path.join(stroke_dir, alphabet_name)

        num_images = sum([len(subfolder) for subfolder in os.listdir(self.img_dir)])
        num_strokes = sum([len(subfolder) for subfolder in os.listdir(self.stroke_dir)])
        assert num_images == num_strokes
        self.dataset_size = num_images
        self.transforms = transforms

    @staticmethod
    def _load_motor(fn):
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

    @staticmethod
    def _load_img(fn):
        I = plt.imread(fn)
        I = np.array(I, dtype=bool)
        return I

    def __getitem__(self, token: str):
        # TODO: encode somehow the trace repetition, right now we always use the first one
        # token_idx, rep_idx = divmod(idx, self.traces_per_char)
        rep_idx = 1

        # text = self.token_to_text[text_token_idx].lower()
        # TODO: why is the space encoded as a 218

        token_images = []
        token_traces = []

        if token == '<|endoftext|>':
            token = " "  # TODO: we are aliasing the space character to end of text

        for char in token:
            character_id = ord(char) - ord('a')

            if character_id == ord(' ') - ord('a'):
                image_so_far, motor_traces = self.char_id_to_sample(ord('a') - ord('a'), rep_idx)
                char_image = np.zeros_like(image_so_far)
                motor_traces = np.zeros_like(motor_traces)
            else:
                char_image, motor_traces = self.char_id_to_sample(character_id, rep_idx)

            assert 0 <= character_id < 26 or char == ' '

            single_channel_image = torch.from_numpy(char_image.copy()).unsqueeze(0)

            single_channel_image, motor_traces = self.transforms(
                images=single_channel_image, motor_traces=motor_traces
            )

            token_images.append(single_channel_image)
            token_traces.append(motor_traces)

        token_images = torch.stack(token_images, dim=0)
        token_traces = torch.stack(token_traces, dim=0)

        return token_images, token_traces

    def char_id_to_sample(self, character_id, rep_idx):
        character_id = f"character{character_id + 1:02d}"
        fn_stk, fn_img = self._get_file_names(character_id, rep_idx)
        motor_traces = self._load_motor(fn_stk)
        resampled_motor_traces = self._resample_traces(motor_traces)
        image = self._load_img(fn_img)
        image_so_far, all_traces = self._process_image_and_traces(image, resampled_motor_traces)
        motor_traces = self._merge_traces(all_traces)
        image_so_far = self._adjust_image_orientation(image_so_far)
        return image_so_far.astype(np.uint8) * 255, motor_traces

    def _get_file_names(self, character_id, rep_idx):
        img_char_dir = os.path.join(self.img_dir, character_id)
        stroke_char_dir = os.path.join(self.stroke_dir, character_id)
        fn_example = os.listdir(img_char_dir)[0]
        fn_base = fn_example[:fn_example.find('_')]
        fn_stk = os.path.join(stroke_char_dir, f"{fn_base}_{rep_idx:02d}.txt")
        fn_img = os.path.join(img_char_dir, f"{fn_base}_{rep_idx:02d}.png")
        return fn_stk, fn_img

    def _resample_traces(self, motor_traces):
        return [
            resample_stroke(stroke, num_samples=constants.POINTS_IN_MOTOR_SEQUENCE // len(motor_traces)) for stroke in
            motor_traces
        ]

    def _process_image_and_traces(self, image, resampled_motor_traces):
        image_so_far = np.zeros_like(image)
        all_traces = np.concatenate(resampled_motor_traces, axis=0)
        min_x, min_y = np.min(all_traces, axis=0)
        max_x, max_y = np.max(all_traces, axis=0)
        assert image_so_far.shape[0] == image_so_far.shape[1]
        for trace in resampled_motor_traces:
            stroke = normalize_trace(trace, min_x, max_x, min_y, max_y)
            points = np.round(stroke[:, :2] * image_so_far.shape[0]).astype(int)
            points = np.clip(points, 0, image_so_far.shape[0] - 1)
            image_so_far[points[:, 1], points[:, 0]] = 1
        return image_so_far, all_traces

    def _merge_traces(self, all_traces):
        motor_traces = np.zeros((constants.POINTS_IN_MOTOR_SEQUENCE, 2))
        motor_traces_ = np.array(all_traces, dtype=np.float32)
        motor_traces[-len(motor_traces_):] = motor_traces_
        return motor_traces

    def _adjust_image_orientation(self, image_so_far):
        image_so_far = np.rot90(image_so_far, k=2)
        image_so_far = np.fliplr(image_so_far)
        return image_so_far

    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':
    # Example Usage
    img_dir = '/home/delverme/Downloads/images_background_small1'
    stroke_dir = '/home/delverme/Downloads/strokes_background_small1/strokes_background_small1'
    # alphabet_names = [a for a in os.listdir(img_dir) if a[0] != '.']
    alphabet_names = ["Latin", ]

    dataset = OmniglotDataset(img_dir, stroke_dir, alphabet_names)
    sample = dataset[0]  # get a sample
    image_so_far, trace, prev_chars = sample


def clean_char(char):
    char = char.lower()
    if char.isascii() and char.isalpha():
        return char
    else:
        return ' '


def clean_token(token):
    if token.startswith('<|') and token.endswith('|>'):
        return token
    text = ''.join([clean_char(c) for c in token])
    return text


class TextTraceDataset(Dataset):
    def __init__(self, omniglot_dataset, text_dataset, tokenizer, token_to_text):
        super().__init__()
        self.omniglot_dataset = omniglot_dataset
        self.text_dataset = text_dataset
        self.tokenizer = tokenizer
        self.token_to_text = token_to_text

    def __len__(self):
        return min(len(self.omniglot_dataset), len(self.text_dataset))

    def __getitem__(self, idx):
        print(self.text_dataset[idx])

        sentence_to_encode = self.text_dataset[idx]
        encoded_text = self.tokenizer.encode_plus(
            sentence_to_encode, truncation=True, max_length=constants.TOKEN_CONTEXT_LEN, padding='max_length',
            return_tensors='pt'
        )

        token_ids, = encoded_text.data['input_ids']
        encoding, = encoded_text.encodings

        tokens = encoding.tokens
        tokens = [clean_token(t) for t in tokens]

        images = []
        text_so_far = []

        motor_contexts = []
        text_contexts = []

        for token_idx, token in zip(token_ids, tokens):
            token_images, token_motor_traces = self.omniglot_dataset[token]

            # token_images, token_motor_traces = self.transforms(
            #     images=token_images,
            #     motor_traces=token_motor_traces
            # )

            char_context = np.array(text_so_far)
            assert constants.TEXT_PADDING_ID not in char_context

            left_padded_char_context = np.pad(
                char_context, (constants.TOKEN_CONTEXT_LEN - len(char_context), 0), 'constant',
                constant_values=constants.TEXT_PADDING_ID)

            left_padded_images = torch.zeros(constants.CHARS_PER_TOKEN, *token_images.shape[1:])
            left_padded_images[-len(token_images):] = token_images

            left_padded_motor_traces = torch.zeros(constants.CHARS_PER_TOKEN, *token_motor_traces.shape[1:])
            left_padded_motor_traces[-len(token_motor_traces):] = token_motor_traces

            text_so_far.append(token_idx)

            images.append(left_padded_images)
            motor_contexts.append(left_padded_motor_traces)
            text_contexts.append(left_padded_char_context)

        token_context_ids = torch.tensor(np.array(text_contexts), dtype=torch.long)
        batch = DataSample(
            images=torch.stack(images, dim=0),
            motor_context=torch.stack(motor_contexts, dim=0),
            text_context_ids=token_context_ids,
            labels=torch.tensor(text_so_far, dtype=torch.long),
        )
        return dataclasses.asdict(batch)

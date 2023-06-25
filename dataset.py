import dataclasses
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

import constants
from utils import DataSample


def resample_stroke(stroke, num_samples=100):
    x, y, t = stroke.T
    new_t = np.linspace(t[0], t[-1], num_samples)
    return np.stack([np.interp(new_t, t, x), np.interp(new_t, t, y)], axis=1)


def normalize_trace(trace, min_x, max_x, min_y, max_y):
    return np.stack([(trace[:, 0] - min_x) / (max_x - min_x), (trace[:, 1] - min_y) / (max_y - min_y)], axis=1)


class MultimodalTransform:
    def __init__(self, image_transform, trace_transform):
        self.image_transform = image_transform
        self.trace_transform = trace_transform

    def __call__(self, images, motor_traces):
        return self.image_transform(images), self.trace_transform(motor_traces)


class OmniglotDataset(Dataset):
    def __init__(self, img_dir: str, stroke_dir: str, transforms: MultimodalTransform, alphabet_name: str = "Latin"):
        self.img_dir = os.path.join(img_dir, alphabet_name)
        self.stroke_dir = os.path.join(stroke_dir, alphabet_name)
        self.dataset_size = self._calculate_dataset_size()
        self.transforms = transforms

    def _calculate_dataset_size(self) -> int:
        num_images = sum([len(subfolder) for subfolder in os.listdir(self.img_dir)])
        num_strokes = sum([len(subfolder) for subfolder in os.listdir(self.stroke_dir)])
        assert num_images == num_strokes
        return num_images

    def _load_motor(self, fn: str) -> np.ndarray:
        with open(fn, 'r') as fid:
            lines = [l.strip() for l in fid.readlines()]
        motor = []
        stk = []
        for myline in lines:
            if myline in ['START', 'BREAK']:
                if stk:
                    motor.append(np.array(stk))
                    stk = []
            else:
                stk.append(np.fromstring(myline, dtype=float, sep=','))
        return motor

    @staticmethod
    def _load_img(fn: str) -> np.ndarray:
        return np.array(plt.imread(fn), dtype=bool)

    def __getitem__(self, token: str):
        # TODO: encode somehow the trace repetition, right now we always use the first one
        # token_idx, rep_idx = divmod(idx, self.traces_per_char)
        rep_idx = 1

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
    def __init__(self, omniglot_dataset, text_dataset, tokenizer):
        super().__init__()
        self.omniglot_dataset = omniglot_dataset
        self.text_dataset = text_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return min(len(self.omniglot_dataset), len(self.text_dataset))

    def __getitem__(self, idx):
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

            char_context = np.array(text_so_far)
            assert constants.TEXT_PADDING_ID not in char_context

            left_padded_char_context = np.pad(
                char_context, (constants.TOKEN_CONTEXT_LEN - len(char_context), 0), 'constant',
                constant_values=constants.TEXT_PADDING_ID)

            left_padded_images = torch.zeros(constants.MAX_CHARS_PER_TOKEN, *token_images.shape[1:])
            left_padded_images[-len(token_images):] = token_images

            left_padded_motor_traces = torch.zeros(constants.MAX_CHARS_PER_TOKEN, *token_motor_traces.shape[1:])
            left_padded_motor_traces[-len(token_motor_traces):] = token_motor_traces

            # TODO: the mod is temporary
            text_so_far.append(token_idx % constants.VOCAB_SIZE)

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

import dataclasses
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import transformers
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import PreTrainedTokenizer, BatchEncoding

import constants
import hyper
from presets import get_default_tokenizer


def cache_dataset():
    if os.path.exists(constants.TEXT_DATASET_PATH):
        print("Dataset already cached")
        return

    response = requests.get(constants.DATA_URL)
    response.raise_for_status()

    with open(constants.TEXT_DATASET_PATH, 'w') as f:
        f.write(response.text)


def resample_stroke(stroke, num_samples=100):
    x, y, t = stroke.T
    new_t = np.linspace(t[0], t[-1], num_samples)
    return np.stack([np.interp(new_t, t, x), np.interp(new_t, t, y)], axis=1)


def normalize_trace(trace, min_x, max_x, min_y, max_y):
    x_norm = max_x - min_x
    if x_norm == 0:
        x_norm = 1
    y_norm = max_y - min_y
    if y_norm == 0:
        y_norm = 1
    return np.stack([
        (trace[:, 0] - min_x) / x_norm,
        (trace[:, 1] - min_y) / y_norm,
    ], axis=1)


@dataclasses.dataclass
class DataSpec:
    use_images: bool
    use_motor_traces: bool
    points_in_motor_sequence: int = hyper.POINTS_IN_MOTOR_SEQUENCE
    image_size: int = hyper.IMAGE_SIZE


class MultimodalTransform:
    def __init__(self, image_transform, trace_transform):
        self.image_transform = image_transform
        self.trace_transform = trace_transform

    def __call__(self, images, motor_traces):
        return self.image_transform(images), self.trace_transform(motor_traces)


class OmniglotDataset(Dataset):
    def __init__(self, data_spec: DataSpec, transforms: MultimodalTransform, alphabet_name: str = "Latin"):
        self.use_images = data_spec.use_images
        self.use_motor_traces = data_spec.use_motor_traces

        self.img_dir = os.path.join(constants.IMG_PATH, alphabet_name)
        self.stroke_dir = os.path.join(constants.TRACES_PATH, alphabet_name)
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

        if not (self.use_images or self.use_motor_traces):
            return None, None

        if token == '<|endoftext|>':
            token = " "  # TODO: we are aliasing the space character to end of text

        for char in token:
            character_id = ord(char) - ord('a')
            assert 0 <= character_id < 26 or char == ' '

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

        if not self.use_images:
            token_images = None
        if not self.use_motor_traces:
            token_traces = None

        return token_images, token_traces

    def char_id_to_sample(self, character_id, rep_idx):
        character_id_str = f"character{character_id + 1:02d}"
        fn_stk, fn_img = self._get_file_names(character_id_str, rep_idx)
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
            resample_stroke(
                stroke, num_samples=hyper.POINTS_IN_MOTOR_SEQUENCE // len(motor_traces)) for stroke in motor_traces
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
        motor_traces = np.zeros((hyper.POINTS_IN_MOTOR_SEQUENCE, 2))
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
    if text == '':
        text = " "
    return text.lower()


class MergeDatasets(Dataset):
    def __init__(self, omniglot_dataset: OmniglotDataset, text_dataset: List[str], tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.omniglot_dataset = omniglot_dataset
        self.text_dataset = text_dataset
        self.tokenizer = tokenizer
        self.empty_token, = self.tokenizer.encode(constants.EMPTY_CHAR, add_special_tokens=False)

    def __len__(self):
        return min(len(self.omniglot_dataset), len(self.text_dataset))

    @torch.no_grad()
    def __getitem__(self, idx):
        # Does not return an item but rather a decomposition of a sentence into several items
        sentence = self.text_dataset[idx]

        images = []
        text_so_far = []

        motor_contexts = []
        text_contexts = []
        sentence_tokens = sentence["input_ids"].tolist()
        # assert len(sentence_tokens) <= constants.MAX_CHARS_PER_TOKEN, "too many characters in a single token"

        # reversed_sentence = self.tokenizer.decode(sentence_tokens, clean_up_tokenization_spaces=True)
        # tokenized_target = self.tokenizer.tokenize(reversed_sentence)

        for token_idx in sentence_tokens:
            token = self.tokenizer.decode(token_idx, clean_up_tokenization_spaces=True).lower()
            token = clean_token(token)

            token_images, token_motor_traces = self.omniglot_dataset[token]

            char_context = np.array(text_so_far)

            assert len(char_context) <= hyper.TOKEN_CONTEXT_LEN, "not enough context to represent the full token"
            left_padded_char_context = np.pad(
                char_context, (hyper.TOKEN_CONTEXT_LEN - len(char_context), 0), 'constant',
                constant_values=self.tokenizer.pad_token_id)

            if token_images is not None:
                assert len(token_images) <= constants.MAX_CHARS_PER_TOKEN, "too many images for a single token"
                left_padded_images = torch.zeros(constants.MAX_CHARS_PER_TOKEN, *token_images.shape[1:])
                left_padded_images[-len(token_images):] = token_images
                images.append(left_padded_images)

            if token_motor_traces is not None:
                left_padded_motor_traces = torch.zeros(constants.MAX_CHARS_PER_TOKEN, *token_motor_traces.shape[1:])
                left_padded_motor_traces[-len(token_motor_traces):] = token_motor_traces
                motor_contexts.append(left_padded_motor_traces)

            text_so_far.append(token_idx)

            if constants.TEXT_PADDING_ID in text_so_far:
                print("Found padding token in text so far, returning another sample")
                return self[idx + 1]

            text_contexts.append(left_padded_char_context)

        batch = DataSample(
            image_context=torch.stack(images, dim=0) if len(images) > 0 else None,
            motor_context=torch.stack(motor_contexts, dim=0) if len(motor_contexts) > 0 else None,
            token_context=torch.from_numpy(np.array(text_contexts)).to(dtype=torch.long),
            labels=torch.tensor(text_so_far, dtype=torch.long),
        )
        return dataclasses.asdict(batch)


class MemoryCachedMergedDataset(MergeDatasets):
    def __init__(self, omniglot_dataset, text_dataset, token_map):
        super().__init__(omniglot_dataset, text_dataset, token_map)
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def __getitem__(self, idx):
        if self.hits + self.misses > 0:
            print("Hits:", self.hits, "Misses:", self.misses, "Hit Rate:", self.hits / (self.hits + self.misses))
        if idx in self.cache:
            self.hits += 1
            return self.cache[idx]
        else:
            self.misses += 1
            sample = super().__getitem__(idx)
            self.cache[idx] = sample
            print("Cache size in MB:", calc_size(self.cache) / 1024 / 1024)
            return sample


def calc_size(obj):
    size = 0
    for k, v in obj.items():
        if isinstance(v, DataSample):
            size += calc_size(dataclasses.asdict(v))
        elif isinstance(v, dict):
            size += calc_size(v)
        else:
            size += v.element_size() * v.nelement()
    return size


class LineByLineTextDataset(transformers.LineByLineTextDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{
            "input_ids": torch.tensor(e, dtype=torch.long),
            "text": l

        } for e, l in zip(self.examples, lines)]


def get_text_dataset(tokenizer):
    cache_dataset()
    text_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=constants.TEXT_DATASET_PATH,
        block_size=128
    )
    if constants.DATASET_SIZE is not None:
        assert len(text_dataset) >= constants.DATASET_SIZE
        text_dataset.examples = text_dataset.examples[:constants.DATASET_SIZE]
        print("Dataset size:", len(text_dataset))
    train_dataset, val_dataset = train_test_split(text_dataset, test_size=0.2)
    return train_dataset, val_dataset


def get_multimodal_dataset(data_spec):
    multimodal_transforms = MultimodalTransform(
        image_transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((data_spec.image_size, data_spec.image_size)),
            transforms.ToTensor()
        ]),
        trace_transform=transforms.Lambda(lambda x: torch.Tensor(x))
    )
    tokenizer = get_default_tokenizer()

    text_test_set, text_train_set = get_text_dataset(tokenizer)

    train_set = MergeDatasets(
        OmniglotDataset(data_spec, transforms=multimodal_transforms),
        text_train_set,
        tokenizer=tokenizer,
    )
    test_set = MergeDatasets(
        OmniglotDataset(data_spec, transforms=multimodal_transforms),
        text_test_set,
        tokenizer=tokenizer,
    )
    return train_set, test_set


@dataclasses.dataclass
class DataSample:
    token_context: BatchEncoding
    labels: Optional[torch.Tensor]
    image_context: Optional[torch.Tensor] = None
    motor_context: Optional[torch.Tensor] = None
    # next_token_logits: Optional[torch.Tensor]

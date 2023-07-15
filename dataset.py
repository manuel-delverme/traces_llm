import dataclasses
import math
import os
import random
import zipfile
from typing import List, Optional, Tuple

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


def cache_text_dataset():
    if os.path.exists(constants.TEXT_DATASET_PATH):
        print("Dataset already cached")
        return

    response = requests.get(constants.DATA_URL)
    response.raise_for_status()

    with open(constants.TEXT_DATASET_PATH, 'w') as f:
        f.write(response.text)


def resample_stroke(stroke, num_samples):
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
    image_side: int = hyper.IMAGE_SIDE
    image_channels: int = 1


class MultimodalTransform:
    def __init__(self, image_transform, trace_transform):
        self.image_transform = image_transform
        self.trace_transform = trace_transform

    def __call__(self, images, motor_traces):
        return self.image_transform(images), self.trace_transform(motor_traces)


# def cache_omniglot_dataset(alphabet_name: str):
#     img_dir = os.path.join(constants.IMG_PATH, alphabet_name)
#     stroke_dir = os.path.join(constants.TRACES_PATH, alphabet_name)
#
#     if not os.path.exists(img_dir):
#         # Download from:
#         # https://github.com/brendenlake/omniglot

def cache_omniglot_dataset():
    maybe_download("images", constants.IMG_PATH)
    maybe_download("strokes", constants.TRACES_PATH)


def maybe_download(file_name: str, path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        strokes_url = f"https://raw.githubusercontent.com/brendenlake/omniglot/master/python/{file_name}_background.zip"
        r = requests.get(strokes_url)
        tmp_file = os.path.join(path, f"{file_name}.zip")
        with open(tmp_file, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(tmp_file, 'r') as zip_ref:
            zip_ref.extractall(path)


def get_omniglot_dataset(data_spec: DataSpec, transforms: MultimodalTransform):
    cache_omniglot_dataset()
    train_dataset = OmniglotDataset(data_spec, transforms, repetitions=tuple(range(1, 18)))
    test_dataset = OmniglotDataset(data_spec, transforms, repetitions=tuple(range(18, 21)))
    return train_dataset, test_dataset


def resample_storkes(motor_traces):
    return [
        resample_stroke(
            stroke, num_samples=hyper.POINTS_IN_MOTOR_SEQUENCE // len(motor_traces)) for stroke in motor_traces
    ]


def _process_image_and_traces(image, resampled_motor_traces):
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


def strokes_to_trace(all_traces):
    motor_traces = np.zeros((hyper.POINTS_IN_MOTOR_SEQUENCE, 2))
    motor_traces_ = np.array(all_traces, dtype=np.float32)
    motor_traces[-len(motor_traces_):] = motor_traces_
    return motor_traces


def _adjust_image_orientation(image_so_far):
    image_so_far = np.rot90(image_so_far, k=2)
    image_so_far = np.fliplr(image_so_far)
    return image_so_far


def postprocess_image_and_traces(image, strokes_for_char):
    resampled_motor_traces = resample_storkes(strokes_for_char)
    image_so_far, all_traces = _process_image_and_traces(image, resampled_motor_traces)
    motor_trace = strokes_to_trace(all_traces)
    image_so_far = postprocess_omniglot_image(image_so_far)
    return image_so_far, motor_trace


def postprocess_omniglot_image(image):
    image_so_far = _adjust_image_orientation(image)
    return image_so_far.astype(np.uint8) * 255


class OmniglotDataset(Dataset):
    def __init__(self, data_spec: DataSpec, transforms: MultimodalTransform, repetitions: Tuple[int, ...]):
        assert max(repetitions) < 20

        alphabet_name = "Latin"
        self.use_images = data_spec.use_images
        self.use_motor_traces = data_spec.use_motor_traces
        self.img_dir = os.path.join(constants.IMG_PATH, "images_background", alphabet_name)
        self.stroke_dir = os.path.join(constants.TRACES_PATH, "strokes_background", alphabet_name)
        self.dataset_size = self._calculate_dataset_size()
        self.transforms = transforms
        self.repetitions = repetitions

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
        # rep_idx = 1
        rep_idx = random.choice(self.repetitions)

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
                # Get a random image and trace
                char_image_raw, motor_traces_raw = self.char_id_to_sample(character_id=0, rep_idx=1)
                char_image, motor_traces = postprocess_image_and_traces(char_image_raw, motor_traces_raw)
                # Set the image and trace to zero
                char_image[:] = 0
                motor_traces[:] = 0
            else:
                char_image_raw, motor_traces_raw = self.char_id_to_sample(character_id, rep_idx)
                char_image, motor_traces = postprocess_image_and_traces(char_image_raw, motor_traces_raw)

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
        image = self._load_img(fn_img)
        return image, motor_traces

    def _get_file_names(self, character_id, rep_idx):
        img_char_dir = os.path.join(self.img_dir, character_id)
        stroke_char_dir = os.path.join(self.stroke_dir, character_id)
        fn_example = os.listdir(img_char_dir)[0]
        fn_base = fn_example[:fn_example.find('_')]
        fn_stk = os.path.join(stroke_char_dir, f"{fn_base}_{rep_idx:02d}.txt")
        fn_img = os.path.join(img_char_dir, f"{fn_base}_{rep_idx:02d}.png")
        return fn_stk, fn_img

    def __len__(self):
        return self.dataset_size


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


def pad_motor_trace(token_motor_traces: torch.Tensor, eager_rate=1.):
    assert token_motor_traces.shape[1:] == (hyper.POINTS_IN_MOTOR_SEQUENCE, 2)
    left_padded_motor_traces = torch.zeros(constants.MAX_CHARS_PER_TOKEN, *token_motor_traces.shape[1:])
    # Calculate the number of characters to keep
    num_chars_to_keep = len(token_motor_traces) * eager_rate
    # If the number of characters to keep is not an integer or is zero, adjust it accordingly
    has_fraction, integer_part = math.modf(num_chars_to_keep)
    should_adjust_last = bool(has_fraction) or integer_part == 0
    num_chars_to_keep = int(num_chars_to_keep) if not should_adjust_last else int(num_chars_to_keep) + 1
    # Slice the motor traces array from the end
    token_motor_traces = token_motor_traces[-num_chars_to_keep:]
    # If adjustment is needed, zero out corresponding steps in the last trace
    if should_adjust_last:
        steps_to_zero = int(has_fraction * token_motor_traces.shape[1])
        token_motor_traces[-1, -steps_to_zero:] = 0
    left_padded_motor_traces[-len(token_motor_traces):] = token_motor_traces
    return left_padded_motor_traces


class MergeDatasets(Dataset):
    def __init__(self, omniglot_dataset: OmniglotDataset, text_dataset: List[str], tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.omniglot_dataset = omniglot_dataset
        self.text_dataset = text_dataset
        self.tokenizer = tokenizer
        self.empty_token, = self.tokenizer.encode(constants.EMPTY_CHAR, add_special_tokens=False)

    def __len__(self):
        return len(self.text_dataset)

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

        # decoded_sentence = self.tokenizer.decode(sentence_tokens, clean_up_tokenization_spaces=True)
        # tokenized_sentence = self.tokenizer.tokenize(decoded_sentence)

        for token_idx in sentence_tokens:
            token = self.tokenizer.decode(token_idx, clean_up_tokenization_spaces=True).lower()
            token = clean_token(token)

            token_images, token_motor_traces = self.omniglot_dataset[token]

            char_context = np.array(text_so_far)

            if len(char_context) > hyper.TOKEN_CONTEXT_LEN:
                char_context = char_context[-hyper.TOKEN_CONTEXT_LEN:]
            left_padded_char_context = np.pad(
                char_context, (hyper.TOKEN_CONTEXT_LEN - len(char_context), 0), 'constant',
                constant_values=self.tokenizer.pad_token_id)

            if token_images is not None:
                assert len(token_images) <= constants.MAX_CHARS_PER_TOKEN, "too many images for a single token"
                left_padded_images = torch.zeros(constants.MAX_CHARS_PER_TOKEN, *token_images.shape[1:])
                left_padded_images[-len(token_images):] = token_images
                images.append(left_padded_images)

            if token_motor_traces is not None:
                if constants.eager_rate < 1:
                    assert token_images is None, "eager rate is not supported for images, as it would leak information"
                left_padded_motor_traces = pad_motor_trace(token_motor_traces, eager_rate=constants.eager_rate)

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
    cache_text_dataset()
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
            transforms.Resize((data_spec.image_side, data_spec.image_side)),
            transforms.ToTensor()
        ]),
        trace_transform=transforms.Lambda(lambda x: torch.Tensor(x))
    )
    tokenizer = get_default_tokenizer()

    text_test_set, text_train_set = get_text_dataset(tokenizer)

    # TODO: we are using the same data for train and test, we should split omniglot in two and sample independently
    train_omniglot_dataset, test_omniglot_dataset = get_omniglot_dataset(data_spec, transforms=multimodal_transforms)

    train_set = MergeDatasets(
        train_omniglot_dataset,
        text_train_set,
        tokenizer=tokenizer,
    )
    test_set = MergeDatasets(
        test_omniglot_dataset,
        text_test_set,
        tokenizer=tokenizer,
    )
    return train_set, test_set


@dataclasses.dataclass
class DataSample:
    token_context: BatchEncoding
    labels: Optional[torch.Tensor] = None
    image_context: Optional[torch.Tensor] = None
    motor_context: Optional[torch.Tensor] = None
    # next_token_logits: Optional[torch.Tensor]


if __name__ == "__main__":
    data_spec = DataSpec(
        use_images=True,
        use_motor_traces=True,
    )
    _train_dataset, valid_dataset = get_multimodal_dataset(data_spec)
    print("Train dataset size:", len(_train_dataset))
    print("Valid dataset size:", len(valid_dataset))
    print("Train dataset sample:", _train_dataset[0])
    print("Valid dataset sample:", valid_dataset[0])

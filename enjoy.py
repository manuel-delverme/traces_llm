from collections import deque

import torch
from transformers import BatchEncoding

import constants
import dataset
import hyper
import presets
import user_model
import mocks
from dataset import DataSpec, DataSample
from text_only_baseline import GPT2FineTuning


class HandwritingRecognizer:
    def __init__(self, model: GPT2FineTuning):
        self.tokenizer = presets.get_default_tokenizer()
        pad_token = self.tokenizer.pad_token_id

        self.context_window = deque(maxlen=constants.MAX_CHARS_PER_TOKEN)
        self.token_history = deque(maxlen=hyper.TOKEN_CONTEXT_LEN, iterable=[pad_token] * hyper.TOKEN_CONTEXT_LEN)

        self.model = model
        self.model.eval()
        # TODO: use the tokenizer from the model
        self.data_spec = model.data_spec

    def update_history_and_preprocess(self, char_trace):
        assert char_trace.shape[0] == 1, "During evaluation we assume one stroke is one char."
        char_trace = char_trace.squeeze(0)

        stroke = self.resample_stroke(char_trace)
        stroke = torch.tensor(stroke, dtype=torch.float32).unsqueeze(0)

        self.context_window.append(stroke)

        context = torch.concat(list(self.context_window))
        motor_trace = dataset.pad_motor_trace(context, eager_rate=1.)
        return motor_trace

    def resample_stroke(self, motor_context):
        motor_context = dataset.resample_stroke(motor_context, self.data_spec.points_in_motor_sequence)
        return motor_context

    @torch.no_grad()
    def update_history_and_predict(self, mouse_positions):
        txt_context = torch.tensor(list(self.token_history)).unsqueeze(0)
        token_context = BatchEncoding({
            "input_ids": txt_context,
            "attention_mask": torch.ones((1, hyper.TOKEN_CONTEXT_LEN), dtype=torch.long)
        })
        # TODO: where is the token_history updated?

        motor_context = self.update_history_and_preprocess(mouse_positions).unsqueeze(0)
        postprocessed_sample = DataSample(
            token_context=token_context,
            motor_context=motor_context,
        )
        prediction = self.model(postprocessed_sample)
        token_idx = prediction.argmax(dim=-1).item()
        token = self.tokenizer.decode(token_idx)
        return token

    def next_token(self):
        self.context_window.clear()


# Usage:
# model = GPT2FineTuning(DataSpec(
#    use_images=False,
#    use_motor_traces=True,
# ))
# gui = PygameGUI(model)
# gui.run()


def main():
    # the gui is not good for tokens, validate results by sampling data from the dataset
    data_spec = dataset.DataSpec(
        use_images=False,
        use_motor_traces=True,
    )
    # model = GPT2FineTuning(data_spec)
    # load checkpoint "best_model-v24.ckpt"
    model = GPT2FineTuning.load_from_checkpoint(
        "./best_model-v24.ckpt",
        data_spec=data_spec,
        map_location=torch.device('cpu'),
    )
    model.eval()
    model.freeze()

    data_spec = DataSpec(
        use_images=False,
        use_motor_traces=True,
    )
    _train_dataset, valid_dataset = dataset.get_multimodal_dataset(data_spec)
    presets.tokenizer = valid_dataset.tokenizer

    gui = mocks.MockGUI(
        model,
        user_interaction=user_model.OfflineUserInteraction(
            valid_dataset,
        )
    )
    # gui = PygameGUI(model, user_interaction=user_model.FakeUserInteraction(valid_dataset))

    # for i in range(len(_train_dataset)):
    #     token_images, token_motor_traces = _train_dataset[i]
    #     _train_dataset.visualize_trace(token_motor_traces)
    # prediction = gui.run_once()

    prediction = gui.run_once()
    gui.run()

    print(prediction)


if __name__ == "__main__":
    main()

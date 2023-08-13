import torch

import dataset
import presets
import user_model
from dataset import DataSpec
from gui import PygameGUI
from text_only_baseline import GPT2FineTuning


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
        "./best_model.ckpt",
        data_spec=data_spec,
        map_location=torch.device('cpu'),
    )
    model.eval()
    model.freeze()

    data_spec = DataSpec(
        use_images=False,
        use_motor_traces=True,
    )
    _train_dataset, valid_dataset = dataset.get_multimodal_dataset(data_spec, model.towers)
    presets.tokenizer = valid_dataset.tokenizer

    # gui = mocks.MockGUI(
    #     model,
    #     user_interaction=user_model.OfflineUserInteraction(valid_dataset)
    # )
    gui = PygameGUI(
        model,
        user_interaction=user_model.OfflineUserInteraction(valid_dataset)
    )

    # for i in range(len(_train_dataset)):
    #     token_images, token_motor_traces = _train_dataset[i]
    #     _train_dataset.visualize_trace(token_motor_traces)
    # prediction = gui.run_once()

    # prediction = gui.run_once()
    gui.run()

    # print(prediction)


if __name__ == "__main__":
    main()

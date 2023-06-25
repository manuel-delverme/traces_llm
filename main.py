import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW

import viz
from dataset import OmniglotDataset, TextTraceDataset, MultimodalTransform
from models import MultimodalLLM
from utils import flatten_batch_and_sequence_dims, DataSample


class Trainer:
    def __init__(self, device, model, optimizer):
        self.device = device
        self.model = model
        self.optimizer = optimizer

    def train_epoch(self, data_loader):
        self.model.train()
        for batch in data_loader:
            batch = DataSample(**batch)

            # Images is of shape (batch_size, seq_len, CHARS_PER_TOKEN, image_channels, image_height, image_width)
            batch = flatten_batch_and_sequence_dims(batch, self.device)
            # Images is of shape (batch_size, CHARS_PER_TOKEN, image_channels, image_height, image_width)

            # viz.visualize_one_sample(batch, num_samples=4, tokenizer=data_loader.dataset.tokenizer)
            self._optimize_batch(batch)

    def _optimize_batch(self, batch: DataSample):
        outputs = self.model(batch)
        loss = torch.nn.functional.cross_entropy(outputs, batch.labels)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def validate(self, epoch, epochs, data_loader):
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in data_loader:
                correct, total = self._validate_batch(batch, correct, total)

        print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {correct / total * 100:.2f}%')

    def _validate_batch(self, batch, correct, total):
        batch = DataSample(**batch)
        batch = flatten_batch_and_sequence_dims(batch, self.device)

        outputs = self.model(batch)
        _, predicted = torch.max(outputs, 1)

        total += batch.labels.size(0)
        correct += (predicted == batch.labels).sum().item()
        return correct, total


def fine_tune(trainer, train_set, test_set, epochs=5):
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    for epoch in range(epochs):
        trainer.train_epoch(train_loader)
        trainer.validate(epoch, epochs, test_loader)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    language_model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = MultimodalLLM(language_model).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5)
    trainer = Trainer(device, model, optimizer)

    img_dir = '/home/delverme/Downloads/images_background_small1'
    stroke_dir = '/home/delverme/Downloads/strokes_background_small1/strokes_background_small1'

    multimodal_transforms = MultimodalTransform(
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(x))
        ])
    )

    omniglot_train_set = OmniglotDataset(img_dir, stroke_dir, transforms=multimodal_transforms)
    omniglot_test_set = OmniglotDataset(img_dir, stroke_dir, transforms=multimodal_transforms)

    text_train_set = ["This is the first sentence", "Here is another sentence", "This is yet another sentence"]
    text_test_set = ["This is the first sentence", "Here is another sentence", "This is yet another sentence"]

    train_set = TextTraceDataset(omniglot_train_set, text_train_set, tokenizer,
                                 token_to_text={v: k for k, v in tokenizer.get_vocab().items()})
    test_set = TextTraceDataset(omniglot_test_set, text_test_set, tokenizer,
                                token_to_text={v: k for k, v in tokenizer.get_vocab().items()})

    fine_tune(trainer, train_set, test_set)


if __name__ == '__main__':
    main()

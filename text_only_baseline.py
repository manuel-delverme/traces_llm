import os.path
import sys

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import experiment_buddy
import pytorch_lightning as pl
import requests
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import constants
from constants import VOCAB_SIZE

DATASET_PATH = 'tiny_shakespeare.txt'


def preprocess_labels(labels):
    labels = labels.clone()
    valid = labels != -100
    labels[valid] = labels[valid] % VOCAB_SIZE
    return labels


class OverfitCallback(pl.callbacks.Callback):
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logs = trainer.callback_metrics
        should_stop = logs['val_accuracy'] > 0.99
        trainer.should_stop = trainer.should_stop or bool(should_stop)


class GPT2FineTuning(pl.LightningModule):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
        self.gpt2.requires_grad_(False)

        # Add a linear layer for fine-tuning
        self.linear = torch.nn.Linear(self.gpt2.config.hidden_size, VOCAB_SIZE, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs: CausalLMOutputWithCrossAttentions = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.linear(outputs.hidden_states[-1])
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch.data["input_ids"], batch.data["attention_mask"], batch.data["labels"]

        logits = self(input_ids, attention_mask)

        labels = preprocess_labels(labels)

        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch.data["input_ids"], batch.data["attention_mask"], batch.data["labels"]
        labels = preprocess_labels(labels)

        logits = self(input_ids, attention_mask)
        a = logits.view(-1, logits.size(-1))
        b = labels.view(-1)
        loss = self.loss_fn(a, b)

        _, predicted = torch.max(logits, dim=-1)
        invalid = labels < 0
        valid_predicted = predicted[~invalid]
        valid_labels = labels[~invalid]
        correct = valid_predicted == valid_labels

        accuracy = sum(correct) / correct.numel()

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        # , "val_loss", loss

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def configure_callbacks(self):
        return [
            # EarlyStopping(monitor="val_accuracy", mode="max", patience=10),
            OverfitCallback()
            # ModelCheckpoint(monitor="val_loss"),
        ]

    @staticmethod
    def loss_fn(logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)


def cache_dataset():
    if os.path.exists(DATASET_PATH):
        print("Dataset already cached")
        return

    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    dataset = requests.get(data_url).text

    with open(DATASET_PATH, 'w') as f:
        f.write(dataset)


def main(logger: experiment_buddy.WandbWrapper):
    cache_dataset()

    if constants.DATASET_SIZE <= 100:
        lr = 1e-1
    elif constants.DATASET_SIZE <= 1000:
        lr = 5e-2
    else:
        lr = 2e-5

    model = GPT2FineTuning(learning_rate=lr)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=DATASET_PATH,
        block_size=128
    )
    # dataset.examples = dataset.examples[:int(constants.DOWN_SAMPLE_DATASET_RATIO * len(dataset))]
    dataset.examples = dataset.examples[:constants.DATASET_SIZE]
    print("Dataset size:", len(dataset))

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=0,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer = Trainer(
        max_epochs=-1,
        logger=WandbLogger(experiment=logger.run),
        enable_progress_bar=True,
        log_every_n_steps=1  # len(train_dataloader) - 1
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=train_dataloader,
    )


def buddy_setup():
    experiment_buddy.register_defaults(vars(constants))
    import wandb
    wandb_kwargs = dict(
        monitor_gym=False, entity="delvermm", settings=wandb.Settings(start_method="thread"), save_code=True, mode="offline")
    # esh = ""
    # hostname = ""
    # sweep_config = ""
    # hostname = "cc-beluga"
    # hostname = "cc-cedar"
    # hostname = "mila"
    hostname = ""
    proc_num = 1
    # proc_num = 8
    # sweep_config = "sweep.yaml"
    # proc_num = -1
    sweep_config = ""
    # hostname = "aws://t4g.micro"
    if sys.gettrace() is not None and os.environ.get("BUDDY_DEBUG_DEPLOYMENT") is None:
        hostname = ""
        sweep_config = ""
    esh = """
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=8G
    #SBATCH --time=2:00:00
    #SBATCH --gres=gpu:1
        """.strip() + "\n"
    extra_modules = None
    if hostname == "mila":
        esh += "#SBATCH --partition=main\n"
        extra_modules = [
            "python/3.7",
            "cuda/11.1",
            "pytorch/1.8.1"
        ]
    elif "cc" in hostname:
        esh += "#SBATCH --partition=cpubase_bycore_b4\n"
        esh += "#SBATCH --account=rrg-dprecup\n"
        # esh += "#SBATCH --account=rrg-bengioy-ad\n"
        extra_modules = [
            "python/3.7",
            # "pytorch/1.7", # CC doesn't have pytorch, should be a package
            "cuda/11.1",
            "pytorch/1.8.1"
        ]
    else:
        esh = ""
    # wandb_run_name = f"{hyper.env_name}-{hyper.sync_with_library}"
    tb = experiment_buddy.deploy(
        hostname, wandb_kwargs=wandb_kwargs, extra_slurm_headers=esh, sweep_definition=sweep_config, proc_num=proc_num,
        extra_modules=extra_modules, conda_env="traces_llm"
    )
    return tb


if __name__ == '__main__':
    tb = buddy_setup()
    main(tb)

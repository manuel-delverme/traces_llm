import datetime
import os.path
import socket
import sys
import urllib.parse

import git
import hydra
import omegaconf
import optuna
import pytorch_lightning as pl
import requests
import torch
import wandb
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
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


class GPT2FineTuning(pl.LightningModule):
    def __init__(self, config):
        learning_rate = config.learning_rate
        hidden_size = config.hidden_size
        num_layers = config.num_layers

        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
        self.gpt2.requires_grad_(False)

        # self.head = torch.nn.Linear(self.gpt2.config.hidden_size, VOCAB_SIZE, bias=False)

        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.gpt2.config.hidden_size, hidden_size),

            torch.nn.ReLU(),
            *[torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
            ) for _ in range(num_layers)],

            torch.nn.Linear(hidden_size, VOCAB_SIZE, bias=False),
        )

    def forward(self, input_ids, attention_mask):
        outputs: CausalLMOutputWithCrossAttentions = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.head(outputs.hidden_states[-1])
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
            EarlyStopping(monitor="val_accuracy", mode="max", patience=100),
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


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(hyper: omegaconf.DictConfig):
    cache_dataset()

    wandb_run = wandb.init(
        project=constants.PROJECT_NAME,
        entity=constants.ENTITY,
        config=hyper,
        job_type="train",
    )

    model = GPT2FineTuning(config=hyper.model_config)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=DATASET_PATH,
        block_size=128
    )
    # dataset.examples = dataset.examples[:int(constants.DOWN_SAMPLE_DATASET_RATIO * len(dataset))]
    assert len(dataset) >= constants.DATASET_SIZE
    dataset.examples = dataset.examples[:constants.DATASET_SIZE]
    print("Dataset size:", len(dataset))

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=0,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer = Trainer(
        max_time=datetime.timedelta(hours=1),
        logger=WandbLogger(experiment=wandb_run),
        enable_progress_bar=False,
        log_every_n_steps=50,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=train_dataloader,
    )


def deploy(hostname):
    import experiment_buddy
    experiment_buddy.deploy(url="hydra://mila")
    git_repo = git.Repo(search_parent_directories=True)
    experiment_id = experiment_buddy.ask_experiment_id(hostname, sweep="")
    hash_commit = experiment_buddy.git_sync(experiment_id, git_repo)
    url = urllib.parse.urlparse(f"ssh://{hostname}")
    executor = experiment_buddy.executors.SSHExecutor(url=url)
    executor.setup_remote(extra_slurm_header=None, working_dir=git_repo.working_dir)

    assert __name__ == "__main__"
    entrypoint = os.path.basename(__file__)
    with executor.ssh_session.cd(executor.scripts_folder):
        executor.run(f"python hydra/run_experiment.sh {git_repo.remotes.origin.url} {entrypoint} {hash_commit}")

    # executor.put(os.path.join(git_repo.working_dir, "run_experiment.sh"),
    #              os.path.join(executor.scripts_folder, "run_experiment.sh"))


if __name__ == '__main__':
    if "delverme-" in socket.gethostname():
        deploy("mila")
    else:
        main()

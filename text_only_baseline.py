import datetime
import inspect
import os.path
import sys

import pytorch_lightning as pl
import requests
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import constants
import experiment_buddy
import hyper
from constants import VOCAB_SIZE

DATASET_PATH = 'tiny_shakespeare.txt'


def preprocess_labels(labels):
    labels = labels.clone()
    valid = labels != -100
    labels[valid] = labels[valid] % VOCAB_SIZE
    return labels


class GPT2FineTuning(pl.LightningModule):
    def __init__(self):
        learning_rate = hyper.learning_rate
        hidden_size = hyper.hidden_size
        num_layers = hyper.num_layers

        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.best_validation_loss = float('inf')

        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
        self.gpt2.requires_grad_(False)

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

        if loss < self.best_validation_loss:
            self.best_validation_loss = loss

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        self.log('best_val_loss', self.best_validation_loss, on_epoch=True, prog_bar=True)

        valid_predicted_list = valid_predicted.tolist()
        valid_labels_list = valid_labels.tolist()

        transposed_data = list(map(list, zip(*[valid_predicted_list, valid_labels_list])))
        self.logger.log_text('validation_results', columns=['Predictions', 'Actual Labels'], data=transposed_data)

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


def main(logger: experiment_buddy.WandbWrapper):
    cache_dataset()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=DATASET_PATH,
        block_size=128
    )
    if constants.DATASET_SIZE is not None:
        assert len(dataset) >= constants.DATASET_SIZE
        dataset.examples = dataset.examples[:constants.DATASET_SIZE]
        print("Dataset size:", len(dataset))

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyper.batch_size,
        num_workers=0,
        shuffle=True,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    valid_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hyper.batch_size,
        num_workers=0,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer = Trainer(
        max_time=datetime.timedelta(hours=hyper.training_hours),
        logger=WandbLogger(experiment=logger.run),
        enable_progress_bar=False,
        log_every_n_steps=50,
    )
    model = GPT2FineTuning()
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


def buddy_setup():
    experiment_buddy.register_defaults(vars(hyper))
    import wandb
    wandb_kwargs = dict(
        monitor_gym=False, entity="delvermm", settings=wandb.Settings(start_method="thread"), save_code=True)
    # esh = ""
    # hostname = ""
    sweep_config = ""
    proc_num = 1
    # hostname = "cc-beluga"
    # hostname = "cc-cedar"
    # hostname = "mila"
    hostname = "mila"
    # sweep_config = "sweep.yaml"
    # proc_num = -1
    # hostname = "aws://t4g.micro"
    if sys.gettrace() is not None and os.environ.get("BUDDY_DEBUG_DEPLOYMENT") is None:
        hostname = ""
        sweep_config = ""
    esh = "\n".join(l.strip() for l in """
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=8G
    #SBATCH --time=1:00:00
    #SBATCH --gres=gpu:1
        """.strip().split("\n")
                    ) + "\n"
    extra_modules = None
    if hostname == "mila":
        esh += "#SBATCH --partition=long\n"
        extra_modules = [
            "anaconda/3",
            "cuda/11.1",
            "pytorch/1.8.1"
        ]
    elif "cc" in hostname:
        esh += "#SBATCH --partition=cpubase_bycore_b4\n"
        esh += "#SBATCH --account=rrg-dprecup\n"
        # esh += "#SBATCH --account=rrg-bengioy-ad\n"
        extra_modules = [
            "anaconda/3",
            # "pytorch/1.7", # CC doesn't have pytorch, should be a package
            "cuda/11.1",
            "pytorch/1.8.1"
        ]
    else:
        esh = ""
    has_conda_env_param = inspect.signature(experiment_buddy.deploy).parameters.get("conda_env") is not None
    if has_conda_env_param:
        tb = experiment_buddy.deploy(
            hostname, wandb_kwargs=wandb_kwargs, extra_slurm_headers=esh, sweep_definition=sweep_config,
            proc_num=proc_num,
            extra_modules=extra_modules, conda_env="traces_llm"
        )
    else:
        tb = experiment_buddy.deploy(
            hostname, wandb_kwargs=wandb_kwargs, extra_slurm_headers=esh, sweep_definition=sweep_config,
            proc_num=proc_num,
            extra_modules=extra_modules
        )
    return tb


if __name__ == '__main__':
    tb_ = buddy_setup()
    main(tb_)

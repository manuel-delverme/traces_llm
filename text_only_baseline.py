import os.path
import sys

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


class GPT2FineTuning(pl.LightningModule):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
        self.gpt2.requires_grad_(False)

        # Add a linear layer for fine-tuning
        # self.gpt2.config.vocab_size
        self.linear = torch.nn.Linear(self.gpt2.config.hidden_size, VOCAB_SIZE, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs: CausalLMOutputWithCrossAttentions = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.linear(outputs.hidden_states[-1])
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch.data["input_ids"], batch.data["attention_mask"], batch.data["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch.data["input_ids"], batch.data["attention_mask"], batch.data["labels"]

        logits = self(input_ids, attention_mask)
        a, b = logits.view(-1, logits.size(-1)), labels.view(-1)
        loss = self.loss_fn(a, b)
        self.log('val_loss', loss, prog_bar=True)

        # Calculate accuracy
        _, predicted = torch.max(logits, dim=-1)
        accuracy = torch.sum(predicted == labels) / (labels.shape[0] * labels.shape[1])
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def loss_fn(logits, labels):
        valid = labels != -100
        labels[valid] = labels[valid] % VOCAB_SIZE
        return torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)


def cache_dataset():
    if os.path.exists(DATASET_PATH):
        print("Dataset already cached")
        return

    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    dataset = requests.get(data_url).text

    dataset = dataset[:int(constants.DOWN_SAMPLE_DATASET_RATIO * len(dataset))]
    print("Dataset size:", len(dataset))

    with open(DATASET_PATH, 'w') as f:
        f.write(dataset)


def main(logger: experiment_buddy.WandbWrapper):
    cache_dataset()
    # Initialize your LightningModule
    model = GPT2FineTuning(learning_rate=2e-5)

    # Initialize a trainer
    trainer = Trainer(
        # gpus=1,  # Use one GPU
        max_epochs=5,  # Train for 5 epochs
        logger=WandbLogger(experiment=logger.run)
    )

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=DATASET_PATH, block_size=128)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=train_dataloader,
    )


if __name__ == '__main__':
    # experiment_buddy.register_defaults(vars(constants))
    # host = ""
    # logger = experiment_buddy.deploy()
    # main(logger)

    experiment_buddy.register_defaults(vars(constants))
    import wandb

    wandb_kwargs = dict(
        monitor_gym=False, entity="delvermm", settings=wandb.Settings(start_method="thread"), save_code=True)

    # esh = ""
    # hostname = ""
    # sweep_config = ""
    # hostname = "cc-beluga"
    # hostname = "cc-cedar"
    hostname = "mila"

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
        esh += "#SBATCH --partition=long\n"
    elif "cc" in hostname:
        esh += "#SBATCH --partition=cpubase_bycore_b4\n"
        esh += "#SBATCH --account=rrg-dprecup\n"
        # esh += "#SBATCH --account=rrg-bengioy-ad\n"
        extra_modules = [
            "python/3.7",
            # "pytorch/1.7", # CC doesn't have pytorch, should be a package
        ]
    else:
        esh = ""

    # wandb_run_name = f"{hyper.env_name}-{hyper.sync_with_library}"
    tb = experiment_buddy.deploy(
        hostname, wandb_kwargs=wandb_kwargs, extra_slurm_headers=esh, sweep_definition=sweep_config, proc_num=proc_num,
        extra_modules=extra_modules,
    )
    main(tb)

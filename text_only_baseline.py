import datetime
import inspect
import os.path
import sys

import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import GPT2LMHeadModel, DataCollatorForLanguageModeling
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import dataset
import experiment_buddy
import hyper
from constants import VOCAB_SIZE
from presets import get_default_tokenizer


def get_text_head(input_size, hidden_size, num_layers):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),

        torch.nn.ReLU(),
        *[torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        ) for _ in range(num_layers)],

        torch.nn.Linear(hidden_size, VOCAB_SIZE, bias=False),
    )


def get_motor_head(data_spec, hidden_size, num_layers):
    return torch.nn.Sequential(
        torch.nn.Linear(data_spec.points_in_motor_sequence, hidden_size),
        torch.nn.ReLU(),
        *[torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        ) for _ in range(num_layers)],
        torch.nn.Linear(hidden_size, 2),
    )


def get_image_head(data_spec, hidden_size, num_layers):
    return torch.nn.Sequential(
        torch.nn.Linear(data_spec.image_size, hidden_size),
        torch.nn.ReLU(),
        *[torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        ) for _ in range(num_layers)],
        torch.nn.Linear(hidden_size, 2),
    )


class GPT2FineTuning(pl.LightningModule):
    def __init__(self, data_spec: dataset.DataSpec):
        learning_rate = hyper.learning_rate
        hidden_size = hyper.hidden_size
        num_layers = hyper.num_layers

        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.best_validation_loss = float('inf')

        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
        self.gpt2.requires_grad_(False)

        self.heads = {"text": get_text_head(self.gpt2.config.hidden_size, hidden_size, num_layers)}
        if data_spec.use_motor_traces:
            self.heads["motor"] = get_motor_head(data_spec, hidden_size, num_layers)
        if data_spec.use_images:
            self.heads["image"] = get_image_head(data_spec, hidden_size, num_layers)

    def forward(self, input_ids, attention_mask):
        outputs: CausalLMOutputWithCrossAttentions = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.head(outputs.hidden_states[-1])
        return logits

    def compute_loss(self, logits, labels):
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch.data["input_ids"], batch.data["attention_mask"], batch.data["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch.data["input_ids"], batch.data["attention_mask"], batch.data["labels"]
        logits = self(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels)

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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def configure_callbacks(self):
        return [
            EarlyStopping(monitor="val_accuracy", mode="max", patience=100),
            ModelCheckpoint(monitor="val_loss", mode="min", filename="best_model"),
        ]


def main(logger: experiment_buddy.WandbWrapper):
    tokenizer = get_default_tokenizer()

    train_dataset, valid_dataset = dataset.get_text_dataset(tokenizer=tokenizer)
    data_spec = dataset.DataSpec(
        use_images=False,
        use_motor_traces=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyper.batch_size,
        num_workers=0,
        shuffle=True,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
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
    model = GPT2FineTuning(data_spec)
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

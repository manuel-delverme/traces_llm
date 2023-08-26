import dataclasses
import datetime
import inspect
import os.path
import sys
import tempfile
import collections

import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import GPT2LMHeadModel, BatchEncoding
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import constants
import dataset
import experiment_buddy
import hyper
from constants import VOCAB_SIZE
from dataset import DataSample
from presets import get_default_tokenizer
from utils import calculate_wer
from viz import visualize_one_sample  # noqa


def get_text_head(input_size, hidden_size, num_layers):
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(input_size, hidden_size),

        torch.nn.ReLU(),
        *[torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        ) for _ in range(num_layers)],

        torch.nn.Linear(hidden_size, VOCAB_SIZE, bias=False),
    )


def get_motor_tower(data_spec, hidden_size, num_layers, features_size):
    point_dimensions = 2
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(
            point_dimensions * data_spec.points_in_motor_sequence * constants.MAX_CHARS_PER_TOKEN,
            hidden_size),
        torch.nn.ReLU(),
        *[torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        ) for _ in range(num_layers)],
        torch.nn.Linear(hidden_size, features_size),
    )


class Permute(torch.nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def get_image_tower(data_spec: dataset.DataSpec, hidden_size, num_layers, features_size):
    layers = [
        Permute(0, 2, 1, 3, 4),  # Swap sequence and channel dimensions
        torch.nn.Conv3d(data_spec.image_channels, hidden_size, kernel_size=(3, 3, 3), stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
    ]

    for _ in range(num_layers):
        layers.extend([
            torch.nn.Conv3d(hidden_size, hidden_size, kernel_size=(3, 3, 3), stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        ])

    sequence_length = constants.MAX_CHARS_PER_TOKEN

    # Compute the size of the height and width after all max pooling operations
    side_after_pooling = data_spec.image_side // (2 ** (num_layers + 1))

    feature_map_size = hidden_size * sequence_length * side_after_pooling ** 2

    layers.extend([
        torch.nn.Flatten(),
        torch.nn.Linear(feature_map_size, features_size)
    ])

    return torch.nn.Sequential(*layers)


def timeit(f):
    def timed(*args, **kw):
        self = args[0]
        ts = datetime.datetime.now()
        result = f(*args, **kw)
        te = datetime.datetime.now()
        self.log(f"time/{f.__name__}", (te - ts).total_seconds(), batch_size=1)
        return result

    return timed


class GPT2FineTuning(pl.LightningModule):
    def __init__(self, data_spec: dataset.DataSpec):
        learning_rate = hyper.learning_rate
        hidden_size = hyper.hidden_size
        num_layers = hyper.num_layers

        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.best_validation_loss = float('inf')

        self.towers = torch.nn.ModuleDict({
            "text": GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
        })
        self.towers["text"].requires_grad_(False)
        features_size = self.towers["text"].config.hidden_size

        if data_spec.use_motor_traces:
            self.towers["motor"] = get_motor_tower(data_spec, hidden_size, num_layers, features_size)
            # self.gpt2.config.hidden_size
        if data_spec.use_images:
            self.towers["image"] = get_image_tower(data_spec, hidden_size, num_layers, features_size)

        num_towers = len(self.towers)
        self.head = get_text_head(input_size=features_size * num_towers, hidden_size=hidden_size, num_layers=num_layers)
        self.feats_shape = None
        self.use_text = True
        self.data_spec = data_spec

    # @timeit
    def forward(self, batch: DataSample):
        assert batch.motor_context is None or batch.motor_context.shape[1:] == (
            constants.MAX_CHARS_PER_TOKEN, hyper.POINTS_IN_MOTOR_SEQUENCE, 2)
        features = []
        if batch.token_context is not None:
            if self.use_text or self.feats_shape is None:
                outputs: CausalLMOutputWithCrossAttentions = self.towers["text"](
                    input_ids=batch.token_context.input_ids,
                    attention_mask=batch.token_context.attention_mask,
                )
                last_hidden_state = outputs.hidden_states[-1]
                hidden_state_for_last_token = last_hidden_state[:, -1, :]
                self.feats_shape = hidden_state_for_last_token.shape[1:]

            if not self.use_text:
                hidden_state_for_last_token = torch.zeros(
                    (batch.token_context.input_ids.shape[0],) + self.feats_shape, device=self.device)

            features.append(hidden_state_for_last_token)
        if batch.motor_context is not None:
            features.append(self.towers["motor"](batch.motor_context))
        if batch.image_context is not None:
            features.append(self.towers["image"](batch.image_context))

        features = torch.concat(features, dim=1)
        logits = self.head(features)
        return logits

    def compute_loss(self, logits, labels):
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100)

    # @timeit
    def training_step(self, batch: DataSample, batch_idx):
        batch = DataSample(**{k: v.to(self.device) for k, v in dataclasses.asdict(batch).items() if v is not None})
        logits = self(batch)
        loss = self.compute_loss(logits, batch.labels)
        self.log('train_loss', loss, batch_size=batch.labels.numel())
        return loss

    def autoregressive_prediction(self, batch: DataSample):
        sentence_features, sentence_targets = self.batch_to_sentences(batch)

        wers = []

        for sentence_feature, sentence_target in zip(sentence_features, sentence_targets):
            hypothesis = []
            running_sample = DataSample(
                token_context=BatchEncoding({
                    "input_ids": torch.full(
                        sentence_feature[0].token_context.data["input_ids"].shape,
                        self.tokenizer.pad_token_id),
                    "attention_mask": torch.zeros(sentence_feature[0].token_context.data["attention_mask"].shape),
                }),
                motor_context=None,
            )
            running_predicted_tokens = collections.deque([self.tokenizer.pad_token_id] * hyper.TOKEN_CONTEXT_LEN,
                                                         maxlen=hyper.TOKEN_CONTEXT_LEN)
            running_attention_mask = collections.deque([0] * hyper.TOKEN_CONTEXT_LEN, maxlen=hyper.TOKEN_CONTEXT_LEN)

            for token_idx, sample in enumerate(sentence_feature):
                running_sample.motor_context = sample.motor_context
                attention_mask = running_sample.token_context.data["attention_mask"]
                history = running_sample.token_context.data["input_ids"]

                attention_mask[:, :] = torch.tensor(running_attention_mask)
                history[:, :] = torch.tensor(running_predicted_tokens)

                logits = self(running_sample)

                _, predicted = torch.max(logits, dim=1)

                running_predicted_tokens.append(predicted)

                running_attention_mask.append(1)
                predicted = predicted.squeeze(0)

                hypothesis.append(predicted)

            wer = calculate_wer(sentence_target, hypothesis)
            wers.append(wer)
            # print("TARGET:", [self.tokenizer.decode([t]) for t in sentence_target])
            # print("HYPOTHESIS:", [self.tokenizer.decode([h]) for h in hypothesis])
            # print("WER:", wer)

        return wers

    def batch_to_sentences(self, batch):
        sentence_features = [[], ]
        sentence_targets = []
        last_attention_map_sum = batch.token_context.data["attention_mask"][0].sum()
        for i in range(batch.labels.shape[0]):
            token_context = batch.token_context["input_ids"][i]
            if i > 0 and batch.token_context.data["attention_mask"][i].sum() < last_attention_map_sum:
                non_masked_tokens = last_token_context[last_token_context != self.tokenizer.pad_token_id]
                target_sentence = non_masked_tokens.tolist() + [int(batch.labels[i])]
                sentence_targets.append(target_sentence)
                sentence_features.append([])

            sentence_features[-1].append(
                DataSample(
                    token_context=BatchEncoding({
                        "input_ids": batch.token_context["input_ids"][i].unsqueeze(0),
                        "attention_mask": batch.token_context["attention_mask"][i].unsqueeze(0),
                    }),
                    motor_context=batch.motor_context[i].unsqueeze(0),
                ))

            last_attention_map_sum = batch.token_context.data["attention_mask"][i].sum()
            last_token_context = token_context.clone()
        sentence_targets.append(last_token_context[last_token_context != self.tokenizer.pad_token_id].tolist())
        return sentence_features, sentence_targets

    # @timeit
    def validation_step(self, batch: DataSample, batch_idx):
        batch = DataSample(**{k: v.to(self.device) for k, v in dataclasses.asdict(batch).items() if v is not None})
        # visualize_one_sample(batch, self.tokenizer, num_samples=4)
        logits = self(batch)
        loss = self.compute_loss(logits, batch.labels)

        _, predicted = torch.max(logits, dim=-1)
        invalid = batch.labels < 0
        valid_predicted = predicted[~invalid]
        valid_labels = batch.labels[~invalid]
        correct = valid_predicted == valid_labels

        accuracy = sum(correct) / correct.numel()

        batch_size = batch.labels.numel()

        if loss < self.best_validation_loss:
            self.best_validation_loss = loss

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('best_val_loss', self.best_validation_loss, on_epoch=True, prog_bar=True, batch_size=batch_size)

        wers = self.autoregressive_prediction(batch)
        average_wer = sum(wers) / len(wers)

        # Store and/or log the computed WER
        self.log('val_wer', average_wer)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def configure_callbacks(self):
        # raise "TODO: figure out why motor traces are not normalized for SL"
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        try:
            checkpoint_path = os.path.join(os.environ["HOME"], "scratch", os.environ["SLURM_JOB_ID"], current_time)
        except KeyError:
            checkpoint_path = tempfile.mkdtemp()
        print("Checkpoint path:", checkpoint_path)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        return [
            EarlyStopping(
                monitor=constants.optimized_metric, mode=constants.optimization_mode,
                patience=hyper.early_stopping_patience, verbose=True),
            ModelCheckpoint(
                monitor=constants.optimized_metric, mode=constants.optimization_mode,
                filename="best_model", dirpath=checkpoint_path)
        ]


def main(logger: experiment_buddy.WandbWrapper):
    tokenizer = get_default_tokenizer()

    data_spec = dataset.DataSpec(
        use_images=False,  # True,
        use_motor_traces=True,
    )
    train_dataset, valid_dataset = dataset.get_multimodal_dataset(data_spec)

    num_cpus = 0 if sys.gettrace() else min(os.cpu_count(), 8)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyper.batch_size,
        num_workers=num_cpus,
        shuffle=True,
        # collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        collate_fn=dataset.FlatteningDataCollator(tokenizer),
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=hyper.batch_size,
        num_workers=num_cpus,
        # collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        collate_fn=dataset.FlatteningDataCollator(tokenizer),
    )
    trainer = Trainer(
        max_time=datetime.timedelta(hours=hyper.training_hours),
        logger=WandbLogger(experiment=logger.run),
        enable_progress_bar=False,
        log_every_n_steps=50,
    )
    model = GPT2FineTuning(data_spec)
    model.tokenizer = tokenizer

    sl_checkpoint_path = "best_model.ckpt"

    # TODO: merge the two scripts and models
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=sl_checkpoint_path,
    )
    return model


def buddy_setup():
    experiment_buddy.register_defaults(vars(hyper))
    import wandb
    wandb_kwargs = dict(
        monitor_gym=False, entity="delvermm", settings=wandb.Settings(start_method="thread"), save_code=True)
    # esh = ""
    # hostname = ""
    # sweep_config = ""
    proc_num = 4
    # hostname = "cc-beluga"
    # hostname = "cc-cedar"
    # hostname = "mila"
    hostname = "mila"
    # sweep_config = ""
    sweep_config = "sweep.yaml"
    # proc_num = -1
    # hostname = "aws://t4g.micro"
    if sys.gettrace() is not None and os.environ.get("BUDDY_DEBUG_DEPLOYMENT") is None:
        hostname = ""
        sweep_config = ""
    esh = "\n".join(l.strip() for l in """
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=64G
    #SBATCH --time=12:00:00
    #SBATCH --gres=gpu:32gb:1
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
    # hyper.training_hours = datetime.timedelta(seconds=1).total_seconds() / (60 * 60)
    tb_ = buddy_setup()
    model = main(tb_)
    # for a in train_dataloader:
    #     print(a)

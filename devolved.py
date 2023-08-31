import datetime
import os.path
import sys

import torch
import torch.utils.data
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import dataset
import experiment_buddy
import hyper
import models.trunk
import models.heads
import presets
import utils


def main(logger: experiment_buddy.WandbWrapper):
    tokenizer = presets.get_default_tokenizer()

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
    decoder = models.trunk.GPT2FineTuning(data_spec)
    decoder.tokenizer = tokenizer

    sl_checkpoint_path = None  # "best_model.ckpt"

    # TODO: merge the two scripts and models
    trainer.fit(
        decoder,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=sl_checkpoint_path,
    )
    return decoder


if __name__ == '__main__':
    tb_ = utils.buddy_setup()
    model = main(tb_)

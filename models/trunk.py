import collections
import dataclasses
import datetime
import os
import tempfile

import pytorch_lightning as pl
import torch
import torch.nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from torcheval.metrics import WordErrorRate
from transformers import GPT2LMHeadModel, BatchEncoding
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import constants
import dataset
import hyper
from constants import GPT2_VOCAB_SIZE, VOCAB_SIZE, MAX_CHARS_PER_TOKEN
from dataset import DataSample
from hyper import POINTS_IN_MOTOR_SEQUENCE
from models.heads import get_motor_tower, get_text_head, get_image_tower


class TraceEmbeddingModel(nn.Module):
    def __init__(self, time_steps):
        super(TraceEmbeddingModel, self).__init__()
        self.conv1 = nn.Conv2d(time_steps, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)  # Assumes images are 28x28 pixels
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MultimodalLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.motor_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),  # Input has 2 channels
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * POINTS_IN_MOTOR_SEQUENCE, 128),  # Input features to Linear should match output from Conv1d
            nn.ReLU(),
        )
        self.downscale_vocab = nn.Sequential(
            nn.Linear(GPT2_VOCAB_SIZE, VOCAB_SIZE),
        )

        features_size = (128 + 128) * MAX_CHARS_PER_TOKEN + VOCAB_SIZE
        self.prediction_head = nn.Sequential(
            nn.Linear(features_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, VOCAB_SIZE),
        )

    def forward(self, batch: DataSample):
        # images is a tensor of shape (batch_size, seq_len, 1, 28, 28) reshape to process all images at once
        batch_size, seq_len, _, _, _ = batch.image_context.shape

        flat_images = batch.image_context.flatten(start_dim=0, end_dim=1)
        image_features = self.image_encoder(flat_images)
        image_features = image_features.unflatten(dim=0, sizes=(batch_size, seq_len))

        # motor_context is a tensor of shape (batch_size, MOTOR_CONTEXT_LEN, 2)
        # We are going to feed it into a CNN, so we need to add a channel dimension
        # swap len and 2

        # Transpose to have channels as the second dimension: (batch_size, 2, MOTOR_CONTEXT_LEN)
        flatten_motor_context = batch.motor_context.flatten(start_dim=0, end_dim=1)
        motor_context = flatten_motor_context.transpose(1, 2)

        motor_features = self.motor_encoder(motor_context)

        motor_features = motor_features.unflatten(dim=0, sizes=(batch_size, seq_len, -1))

        # llm_out: CausalLMOutputWithCrossAttentions = self.language_model(batch.text_context_ids)
        # assert llm_out.logits.shape == (batch_size, TOKEN_CONTEXT_LEN, GPT2_VOCAB_SIZE)
        # text_features = llm_out.logits[:, -1, :]  # TODO: check this

        # Logits is a tensor of shape (batch_size, sequence_length, vocab_size)
        # Vocabulary size is ~50k for GPT2, ours is VOCAB_SIZE, so we need to project it down to VOCAB_SIZE
        text_features = self.downscale_vocab(batch.next_token_logits.to(torch.float32))

        # TODO:
        # text has no seq len, it's one token per token
        # cat doesn't work, cat the two tensors, then cat the result with the third tensor
        features = torch.cat((
            image_features.flatten(start_dim=1),
            motor_features.flatten(start_dim=1),
            text_features
        ), dim=1)
        logits = self.prediction_head(features)
        return logits


class GPT2FineTuning(pl.LightningModule):
    def __init__(self, data_spec: dataset.DataSpec):
        learning_rate = hyper.learning_rate
        hidden_size = hyper.hidden_size
        num_layers = hyper.num_layers

        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.skip_first_wer = True

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
        wer_metric = WordErrorRate()
        sentence_features, sentence_targets = self.batch_to_sentences(batch)

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

            # Avid decoding the tokens, it's slow
            wer_metric.update([str(h.item()) for h in hypothesis], [str(t) for t in sentence_target])
        return wer_metric.compute()

    def _append_sentence_features(self, features, token_ids, motor_context):
        features.append(
            DataSample(
                token_context=BatchEncoding({
                    "input_ids": token_ids.unsqueeze(0),
                    "attention_mask": token_ids.ne(self.tokenizer.pad_token_id).unsqueeze(0),
                }),
                motor_context=motor_context.unsqueeze(0),
            )
        )

    def _append_sentence_targets(self, targets, token_ids, label):
        non_masked_tokens = token_ids[token_ids != self.tokenizer.pad_token_id]
        targets.append(non_masked_tokens.tolist() + [int(label)])

    def batch_to_sentences(self, batch):
        sentence_features, sentence_targets = [[]], []
        last_attention_map_sum = batch.token_context.data["attention_mask"][0].sum()

        for i in range(batch.labels.shape[0]):
            token_context = batch.token_context["input_ids"][i]
            attention_map_sum = batch.token_context.data["attention_mask"][i].sum()

            if i > 0 and attention_map_sum < last_attention_map_sum:
                self._append_sentence_targets(sentence_targets, last_token_context, batch.labels[i])
                sentence_features.append([])

            self._append_sentence_features(sentence_features[-1], token_context, batch.motor_context[i])

            last_attention_map_sum = attention_map_sum
            last_token_context = token_context.clone()

        self._append_sentence_targets(sentence_targets, last_token_context, batch.labels[-1])
        return sentence_features, sentence_targets

    # @timeit
    def validation_step(self, batch: DataSample, batch_idx):
        # super().validation_step(batch, batch_idx)

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

        if self.skip_first_wer:
            average_wer = self.autoregressive_prediction(batch)
            self.log('val_wer', average_wer, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.skip_first_wer = False

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

# File: models.py

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from constants import GPT2_VOCAB_SIZE, VOCAB_SIZE, POINTS_IN_MOTOR_SEQUENCE
from utils import DataSample


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
    def __init__(self, language_model: GPT2LMHeadModel):
        super().__init__()
        self.language_model = language_model

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
        self.prediction_head = nn.Sequential(
            nn.Linear(128 + 128 + VOCAB_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, VOCAB_SIZE),
        )

    def forward(self, batch: DataSample):
        image_features = self.image_encoder(batch.images)

        # motor_context is a tensor of shape (batch_size, MOTOR_CONTEXT_LEN, 2)
        # We are going to feed it into a CNN, so we need to add a channel dimension
        # swap len and 2

        # Transpose to have channels as the second dimension: (batch_size, 2, MOTOR_CONTEXT_LEN)
        motor_context = batch.motor_context.transpose(1, 2)

        motor_features = self.motor_encoder(motor_context)

        llm_out: CausalLMOutputWithCrossAttentions = self.language_model(batch.text_context_ids)
        text_features = llm_out.logits[:, -1, :]  # TODO: check this

        # Logits is a tensor of shape (batch_size, sequence_length, vocab_size)
        # Vocabulary size is ~50k for GPT2, ours is VOCAB_SIZE, so we need to project it down to VOCAB_SIZE
        text_features = self.downscale_vocab(text_features)

        features = torch.cat((image_features, motor_features, text_features), dim=1)
        logits = self.prediction_head(features)
        return logits

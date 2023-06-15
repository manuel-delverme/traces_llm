import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import matplotlib.pyplot as plt

from utils import num2str, load_motor


# Include your load_motor, plot_motor_to_image, and num2str functions here


class OmniglotTraceDataset(Dataset):
    def __init__(self, img_dir, stroke_dir, tokenizer, nalpha=5, nreps=20):
        self.img_dir = img_dir
        self.stroke_dir = stroke_dir
        self.tokenizer = tokenizer
        self.nalpha = nalpha
        self.nreps = nreps

        self.alphabet_names = [a for a in os.listdir(img_dir) if a[0] != '.']
        self.alphabet_names = random.sample(self.alphabet_names, nalpha)

    def __len__(self):
        return self.nalpha * self.nreps

    def __getitem__(self, idx):
        alpha_idx = idx // self.nreps
        rep_idx = idx % self.nreps + 1

        alpha_name = self.alphabet_names[alpha_idx]
        character_id = random.randint(1, len(os.listdir(os.path.join(self.img_dir, alpha_name))))

        img_char_dir = os.path.join(self.img_dir, alpha_name, 'character' + num2str(character_id))
        stroke_char_dir = os.path.join(self.stroke_dir, alpha_name, 'character' + num2str(character_id))

        fn_example = os.listdir(img_char_dir)[0]
        fn_base = fn_example[:fn_example.find('_')]

        fn_stk = os.path.join(stroke_char_dir, fn_base + '_' + num2str(rep_idx) + '.txt')
        fn_img = os.path.join(img_char_dir, fn_base + '_' + num2str(rep_idx) + '.png')

        motor = load_motor(fn_stk)
        return motor, fn_img  # return motor and image path, you can also return the processed image tensor


def main():
    img_dir = '/home/delverme/Downloads/images_background_small1'
    stroke_dir = '/home/delverme/Downloads/strokes_background_small1/strokes_background_small1'
    nreps = 20  # number of renditions for each character
    nalpha = 5  # number of alphabets to show

    # Load a pretrained language model and tokenizer
    model_name = "gpt2"  # replace with the name of the model you want to use
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create an instance of the OmniglotTraceDataset class
    dataset = OmniglotTraceDataset(img_dir, stroke_dir, tokenizer, nalpha, nreps)

    # Create a DataLoader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # This is a basic loop to plot images and motor traces, you might need to adapt it for training.
    for motor, img_path in data_loader:
        print(motor.shape, img_path.shape)
        # plot_motor_to_image and other relevant processing functions should be defined
        # plot_motor_to_image(img_path, motor)

        # Additionally, you can perform training here, pass the data to your model etc.
        pass


if __name__ == '__main__':
    main()

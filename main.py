import collections

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW

import constants
import dataset
from dataset import TextTraceDataset
from models import TraceEmbeddingModel, MultimodalLLM
from utils import process_batch, DataSample


def load_text_dataset(file_path):
    # This is a mock implementation, so we will just return a list of tokenized sentences.
    # In the actual implementation, you would read from the file at `file_path` and tokenize the sentences using the tokenizer.

    sentences = [
        "This is the first sentence",
        "Here is another sentence",
        "This is yet another sentence",
        # Add more sentences as needed
    ]
    return sentences


# Define a function to fine-tune the model
def fine_tune(model: MultimodalLLM, train_set, test_set, device, trace_model, epochs=5):
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        train_epoch(device, model, optimizer, train_loader)
        validate(device, epoch, epochs, model, test_loader)


def train_epoch(device, model, optimizer, train_loader):
    model.train()
    for batch in train_loader:
        # Process the batch to get input tokens and labels
        batch = DataSample(**batch)
        input_data_sample, labels = process_batch(batch, device)
        # Input_data_sample is a dictionary with keys "text", "image", and "motor"
        # Images is of shape (batch_size, CHARS_PER_TOKEN, image_channels, image_height, image_width)

        tokenizer = train_loader.dataset.tokenizer
        inverse_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
        # visualize_one_sample(input_data_sample, labels, num_samples=4, token_dict=inverse_vocab)

        outputs = model(input_data_sample)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def validate(device, epoch, epochs, model, test_loader):
    # Validate the model
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = DataSample(**batch)
            # Process the batch to get input tokens and labels
            input_tokens, labels = process_batch(batch, device)

            # Move the input tokens and labels to the device
            input_tokens = input_tokens.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_tokens)
            _, predicted = torch.max(outputs.logits, 1)

            # Update the total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Print the accuracy for this epoch
    print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {correct / total * 100:.2f}%')


def visualize_image(images):
    img_grid = torchvision.utils.make_grid(images, nrow=1, normalize=False, pad_value=1, value_range=(-1, 1))
    img_width = img_grid.shape[2]
    img_height = img_grid.shape[1]
    fig = plt.figure(figsize=(img_width / img_height, 1))
    ax = plt.Axes(fig, [0., 0., 1., 1.], )
    ax.set_axis_off()
    fig.add_axes(ax)
    img_grid = (img_grid - img_grid.min())
    img_grid = img_grid / img_grid.max()
    ax.imshow(img_grid.permute(1, 2, 0), interpolation='none')
    return images


def visualize_motor_context(ax, motor_context, offset):
    context_sequence_len, motor_sequence_length, _ = motor_context.shape

    time_step_indices = np.arange(motor_sequence_length)
    x_cumulative = collections.deque([0, ], maxlen=3)
    y_cumulative = collections.deque([0, ], maxlen=3)

    matplotlib_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    offset = 0

    for context_idx in range(context_sequence_len):
        # Loop through the selected time steps
        color = matplotlib_color_cycle[context_idx % len(matplotlib_color_cycle)]

        token_trace = motor_context[context_idx]
        next_offset = 0

        for seq_index in time_step_indices[1:]:
            # Extract the x and y coordinates at the current time step
            x, y = token_trace[seq_index]
            x_cumulative.append(x.item())
            y_cumulative.append(y.item())

            # Calculate differences for arrow
            dx = x_cumulative[-1] - x_cumulative[-2]
            dy = y_cumulative[-1] - y_cumulative[-2]
            ax.quiver(
                x_cumulative[-2] + offset,
                y_cumulative[-2],
                dx, dy, angles='xy', scale_units='xy', scale=1, color=color)
            next_offset = max(next_offset, x_cumulative[-1])
        offset += next_offset
        print(offset)
    ax.grid()


def visualize_one_sample(input_data_sample: DataSample, labels: torch.Tensor, token_dict, num_samples=4):
    text_context_ids: torch.Tensor = input_data_sample.text_context_ids

    # Create a figure
    fig = plt.figure(figsize=(20, 5 * num_samples))

    for sample_idx in range(num_samples):
        offset = 0
        # Set title as token history
        tokens = text_context_ids[sample_idx].tolist()
        token_history = [token_dict[token] if token_dict else str(token) for token in tokens]
        token_str = " ".join(token_history)
        label_str = token_dict[labels[sample_idx].item()]

        # Replace ! with a symbol, replace non alphanumeric characters with empty string
        def filter_chars(c):
            if c == '!':
                return '␣'
            elif c.isalnum() and c.isascii():
                return c
            elif c == ' ':
                return ' '
            else:
                return ''

        token_str = "".join([filter_chars(c) for c in token_str])
        label_str = "".join([filter_chars(c) for c in label_str])

        # Plot image
        ax_image = fig.add_subplot(num_samples, 2, 2 * sample_idx + 1)
        images = input_data_sample.images[sample_idx]
        nrow = int(np.sqrt(images.shape[0]))
        img_grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=False, pad_value=0, value_range=(-1, 1))
        img_grid -= img_grid.min()
        img_grid /= img_grid.max()
        # Flip black and white
        img_grid = 1 - img_grid
        ax_image.imshow(img_grid.permute(1, 2, 0), interpolation='none')
        ax_image.set_title(token_str + f" => {label_str}", fontsize=12)
        ax_image.axis('off')

        ax_motor = fig.add_subplot(num_samples, 2, 2 * sample_idx + 2)
        motor_context = input_data_sample.motor_context[sample_idx]
        visualize_motor_context(ax_motor, motor_context, offset=offset)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    # Show the plot
    plt.show()


image_transforms = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((224, 224)),  # Resizing to fit models like ResNet
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
    # Standard normalization for models like ResNet
])

trace_transforms = transforms.Compose([
    transforms.Lambda(lambda x: torch.Tensor(x))  # Assuming trace is a numpy array, this converts to a Tensor
    # Add more transformations as necessary
])

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),
    # transforms.Resize((28, 28))  # TODO: remove
])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a pretrained language model and tokenizer
    model_name = "gpt2"  # replace with the name of the model you want to use
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    language_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = MultimodalLLM(language_model).to(device)

    # Load the Omniglot dataset
    # omniglot_train_set = datasets.Omniglot(root='./data', background=True, download=True, transform=transform)
    # omniglot_test_set = datasets.Omniglot(root='./data', background=False, download=True, transform=transform)
    img_dir = '/home/delverme/Downloads/images_background_small1'
    stroke_dir = '/home/delverme/Downloads/strokes_background_small1/strokes_background_small1'

    inverse_vocab = {v: k for k, v in tokenizer.get_vocab().items()}

    multimodal_transforms = dataset.MultimodalTransform(image_transforms, trace_transforms)

    omniglot_train_set = dataset.OmniglotDataset(img_dir, stroke_dir, transforms=multimodal_transforms)
    omniglot_test_set = dataset.OmniglotDataset(img_dir, stroke_dir, transforms=multimodal_transforms)

    text_train_set = load_text_dataset('train.txt')
    text_test_set = load_text_dataset('test.txt')

    # Create instances of the CustomDataset class
    train_set = TextTraceDataset(omniglot_train_set, text_train_set, tokenizer, token_to_text=inverse_vocab)
    test_set = TextTraceDataset(omniglot_test_set, text_test_set, tokenizer, token_to_text=inverse_vocab)

    # Initialize the trace embedding model
    trace_model = TraceEmbeddingModel(constants.TOKEN_CONTEXT_LEN)
    trace_model = trace_model.to(device)

    fine_tune(model, train_set, test_set, device, trace_model)


if __name__ == '__main__':
    main()

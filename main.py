import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW

from models import TraceEmbeddingModel
from utils import process_batch



def load_text_dataset(file_path, tokenizer):
    # This is a mock implementation, so we will just return a list of tokenized sentences.
    # In the actual implementation, you would read from the file at `file_path` and tokenize the sentences using the tokenizer.

    sentences = [
        "This is the first sentence.",
        "Here is another sentence.",
        "This is yet another sentence.",
        # Add more sentences as needed
    ]
    return sentences


class TextTraceDataset(Dataset):
    def __init__(self, omniglot_dataset, text_dataset, tokenizer):
        self.omniglot_dataset = omniglot_dataset
        self.text_dataset = text_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return min(len(self.omniglot_dataset), len(self.text_dataset))

    def __getitem__(self, idx):
        image, label = self.omniglot_dataset[idx]
        text = self.text_dataset[idx]
        text_input = self.tokenizer.encode_plus(
            text, truncation=True, max_length=128, padding='max_length', return_tensors='pt')
        return image, label, text_input


# Define a function to fine-tune the model
def fine_tune(model, train_set, test_set, device, trace_model, tokenizer, epochs=5):
    # Prepare the data loaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # Use the AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)

    # Loop over the epochs
    for epoch in range(epochs):
        # Train the model
        model.train()
        for batch in train_loader:
            # Process the batch to get input tokens and labels
            input_tokens, labels = process_batch(batch, trace_model, device)

            # Move the input tokens and labels to the device
            input_tokens = input_tokens.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_tokens, labels=labels)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validate the model
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                # Process the batch to get input tokens and labels
                input_tokens, labels = process_batch(batch, trace_model)

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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a pretrained language model and tokenizer
    model_name = "gpt2"  # replace with the name of the model you want to use
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Load the Omniglot dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the Omniglot dataset
    omniglot_train_set = datasets.Omniglot(root='./data', background=True, download=True, transform=transform)
    omniglot_test_set = datasets.Omniglot(root='./data', background=False, download=True, transform=transform)

    trace_data = load_motor("path_to_trace_data_file.txt")


    text_train_set = load_text_dataset('train.txt', tokenizer)
    text_test_set = load_text_dataset('test.txt', tokenizer)

    # Create instances of the CustomDataset class
    train_set = TextTraceDataset(omniglot_train_set, text_train_set, train_trace_data, tokenizer)
    test_set = TextTraceDataset(omniglot_test_set, text_test_set, tokenizer)

    # Initialize the trace embedding model
    trace_model = TraceEmbeddingModel()
    trace_model = trace_model.to(device)

    fine_tune(model, train_set, test_set, device, trace_model, tokenizer)


if __name__ == '__main__':
    main()

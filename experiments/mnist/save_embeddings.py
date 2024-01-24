import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import os

from torch.nn.functional import one_hot


def main():
    # Step 1: Download MNIST dataset using PyTorch and split into train and test

    # Define data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),  # Duplicate channel for grayscale image
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
    ])

    # Download MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Step 2: Create data loaders for train and test sets
    batch_size = 64
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Step 3: Download a pretrained ResNet-18 model using PyTorch
    resnet8 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
    # Remove the last fully connected layer (final classification layer)
    modules = list(resnet8.children())[:-1]
    resnet8 = nn.Sequential(*modules)
    resnet8.eval()  # Set the model to evaluation mode

    # Step 4: Get output embeddings from the ResNet-18 model

    def get_embeddings(model, data_loader):
        embeddings, concepts, tasks = [], [], []
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = model(inputs).squeeze()
                outputs_1, outputs_2 = outputs[len(outputs) // 2:], outputs[:len(outputs) // 2]
                concepts_1, concepts_2 = labels[len(labels) // 2:], labels[:len(labels) // 2]
                tasks_i = one_hot(concepts_1 + concepts_2, 19)
                concepts_1, concepts_2 = one_hot(concepts_1, 10), one_hot(concepts_2, 10)
                embeddings.extend(torch.hstack([outputs_1, outputs_2]))
                concepts.extend(torch.hstack([concepts_1, concepts_2]))
                tasks.extend(tasks_i)
        return (torch.stack(embeddings), torch.stack(concepts), torch.stack(tasks))

    train_embeddings = get_embeddings(resnet8, train_loader)
    test_embeddings = get_embeddings(resnet8, test_loader)

    # Step 5: Save embeddings in a file (you can choose the format, e.g., numpy array)
    save_dir = './embeddings'
    os.makedirs(save_dir, exist_ok=True)

    train_embeddings_file = os.path.join(save_dir, 'train_embeddings.pt')
    test_embeddings_file = os.path.join(save_dir, 'test_embeddings.pt')

    torch.save(train_embeddings, train_embeddings_file)
    torch.save(test_embeddings, test_embeddings_file)

    print(f"Train embeddings saved to {train_embeddings_file}")
    print(f"Test embeddings saved to {test_embeddings_file}")


if __name__ == '__main__':
    main()

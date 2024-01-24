import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import os
import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm
import random


from torch.nn.functional import one_hot


def main():
    # Step 1: Download MNIST dataset using PyTorch and split into train and test

    # Define data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(1)),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),  # Duplicate channel for grayscale image
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
    ])

    def transform(dataset_img, dataset_concetp):
        identity = np.eye(3)
        result = []
        new_concepts = []
        for i in range(len(dataset_img)):
            img = dataset_img[i]
            concept = dataset_concetp[i]
            if concept[0] == 1 or concept[1] == 1:
                rand_int = np.random.randint(0, 3)
                color_c = identity[rand_int]
            else:
                rand_int = np.random.randint(0, 2)
                color_c = identity[rand_int]
            new_c = torch.tensor(np.concatenate([concept, color_c]))
            new_concepts += [new_c]
            img = torch.tensor(img).float()
            img = img.unsqueeze(0)
            img = torch.cat([img*color_c[0], img*color_c[1], img*color_c[2]], 0)
            img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
            result += [img]
        new_concepts = torch.stack(new_concepts)
        result = torch.stack(result)
        return result, new_concepts

    # Read DSprites dataset
    save_dir = './datasets/dsprites'
    os.makedirs(save_dir, exist_ok=True)

    train_images_file = os.path.join(save_dir, 'train_images.npy')
    test_images_file = os.path.join(save_dir, 'test_images.npy')
    train_labels_file = os.path.join(save_dir, 'train_labels.npy')
    test_labels_file = os.path.join(save_dir, 'test_labels.npy')
    train_concepts_file = os.path.join(save_dir, 'train_concepts.npy')
    test_concepts_file = os.path.join(save_dir, 'test_concepts.npy')

    train_image, train_concepts = transform(np.load(train_images_file), np.load(train_concepts_file))
    test_image, test_concepts = transform(np.load(test_images_file), np.load(test_concepts_file))
    train_labels = np.load(train_labels_file)
    test_labels = np.load(test_labels_file)

    # Transform to tensor
    train_labels = torch.from_numpy(train_labels).long()
    test_labels = torch.from_numpy(test_labels).long()

    # Step 2: Create data loaders for train and test sets
    batch_size = 64
    train_data = TensorDataset(train_image, train_labels, train_concepts)
    test_data = TensorDataset(test_image, test_labels, test_concepts)

    train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = data_utils.DataLoader(test_data, batch_size=batch_size, shuffle=False)

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
            for inputs, labels, c in tqdm(data_loader):
                outputs = model(inputs).squeeze()
                tasks_i = one_hot(labels, 2)
                # tasks_i = labels
                assert tasks_i.shape[0] == outputs.shape[0]
                embeddings.extend(outputs)
                concepts.extend(c)
                tasks.extend(tasks_i)
        return (torch.stack(embeddings), torch.stack(concepts), torch.stack(tasks))

    train_embeddings = get_embeddings(resnet8, train_loader)
    test_embeddings = get_embeddings(resnet8, test_loader)

    # Step 5: Save embeddings in a file (you can choose the format, e.g., numpy array)
    save_dir = './embeddings/dsprites/'
    os.makedirs(save_dir, exist_ok=True)

    train_embeddings_file = os.path.join(save_dir, 'train_embeddings.pt')
    test_embeddings_file = os.path.join(save_dir, 'test_embeddings.pt')

    torch.save(train_embeddings, train_embeddings_file)
    torch.save(test_embeddings, test_embeddings_file)

    print(f"Train embeddings saved to {train_embeddings_file}")
    print(f"Test embeddings saved to {test_embeddings_file}")


if __name__ == '__main__':
    main()

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import os
import sys
# sys.path.append(
from tqdm import tqdm
from CUB200.cub_loader import load_data

from torch.nn.functional import one_hot


def main():
    root_dir = os.path.join(os.getcwd(), 'experiments/cub/CUB200/class_attr_data_10/')
    train_data_path = os.path.join(root_dir, 'train.pkl')
    val_data_path = os.path.join(root_dir, 'val.pkl')
    test_data_path = os.path.join(root_dir, 'test.pkl')

    batch_size = 64
    
    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=batch_size,
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir=root_dir,
        num_workers=8,
    )

    val_dl = load_data(
        pkl_paths=[val_data_path],
        use_attr=True,
        no_img=False,
        batch_size=batch_size,
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir=root_dir,
        num_workers=8,
    )

    test_dl = load_data(
        pkl_paths=[test_data_path],
        use_attr=True,
        no_img=False,
        batch_size=batch_size,
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir=root_dir,
        num_workers=8,
    )

    # Step 1: Create data loaders for train and test sets    
    train_loader = train_dl
    val_loader = val_dl
    test_loader = test_dl

    # Step 2: Download a pretrained ResNet-18 model using PyTorch
    resnet8 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
    # Remove the last fully connected layer (final classification layer)
    modules = list(resnet8.children())[:-1]
    resnet8 = nn.Sequential(*modules)
    resnet8.eval()  # Set the model to evaluation mode

    # Step 3: Get output embeddings from the ResNet-18 model

    def get_embeddings(model, data_loader_list):
        embeddings, concepts, tasks = [], [], []
        with torch.no_grad():
            for data_loader in data_loader_list:
                for inputs, labels, c in tqdm(data_loader):
                    outputs = model(inputs).squeeze()
                    tasks_i = one_hot(labels, data_loader.dataset.n_classes)
                    assert tasks_i.sum() == outputs.shape[0]
                    embeddings.extend(outputs)
                    concepts.extend(c)
                    tasks.extend(tasks_i)
        return (torch.stack(embeddings), torch.stack(concepts), torch.stack(tasks))

    train_embeddings = get_embeddings(resnet8, [train_loader, val_loader])
    test_embeddings = get_embeddings(resnet8, [test_loader])

    # Step 5: Save embeddings in a file (you can choose the format, e.g., numpy array)
    save_dir = './embeddings/cub/'
    os.makedirs(save_dir, exist_ok=True)

    train_embeddings_file = os.path.join(save_dir, 'train_embeddings.pt')
    test_embeddings_file = os.path.join(save_dir, 'test_embeddings.pt')

    torch.save(train_embeddings, train_embeddings_file)
    torch.save(test_embeddings, test_embeddings_file)

    print(f"Train embeddings saved to {train_embeddings_file}")
    print(f"Test embeddings saved to {test_embeddings_file}")


if __name__ == '__main__':
    main()

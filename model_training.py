from sample import BaseModel, train_predictor, train_selector
from sample import PredictorNetwork
from sample import SelectorNetwork
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time


def train_predictor(model, dataloader, epochs=10):
    print("Training predictor")
    total_start = time.time()
    print("Start time: ", total_start)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    model.train()
    loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
    print(f"total time: {time.time() - total_start}")

    return loss_history

def train_selector(model, dataloader, epochs=10):
    print("Training selector")
    total_start = time.time()
    print("Start time: ", total_start)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    model.train()
    loss_history = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
    print(f"total time: {time.time() - total_start}")

    return loss_history


if __name__ == "__main__":
    """Set up the DataLoaders: """
    # Define the transformation for the validation data
    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load the validation dataset
    print("Loading CIFAR-10 dataset to base model")
    total_start = time.time()
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(f"total time: {time.time() - total_start}")


    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_size = 640 #int(0.1 * len(full_train_dataset)) # 6400
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    test_size = int(0.1 * len(test_dataset))
    _, small_test_dataset = random_split(test_dataset, [len(test_dataset) - test_size, test_size])

    test_loader = DataLoader(small_test_dataset, batch_size=32, shuffle=False)

    # Print the sizes of the datasets
    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(small_test_dataset)}')


    '''Instantiate models, criterion, and optimizer'''
    # base_model = BaseModel()
    base_model = BaseModel()
    base_model.eval()
    num_classes = 10 # CIFAR10

    # Load the pretrained ResNet-50 model
    resnet50 = models.resnet50(pretrained=True) # pretrained on ImageNet
    # resnet50.eval() # sets this to evaluation mode

    resnet18 = models.resnet18(pretrained=True)
    for param in resnet18.parameters():
        param.requires_grad = False
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes) #rewrites resnet18 final fc layer


    '''Build the predictor/selector Database'''

    import torch
    import torch.nn.functional as F
    from torchvision import models, datasets, transforms
    import numpy as np
    from torch.utils.data import DataLoader, Dataset, Subset
    from dataset_utils import PredictorDataset, SelectorDataset

    output_shapes = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet18 = resnet18.to(device)
    resnet18.eval()

    layer1_list, layer2_list, layer3_list, layer4_list, fc_list = [], [], [], [], []
    binary_list = []


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_samples = 640  # Adjust this number as needed
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    val_subset = Subset(val_dataset, indices)
    val_subset_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    print("Collecting Predictor/selector data", train_size, "samples")
    total_start = time.time()
    for images, labels in val_subset_loader: 
        out = images
        # print(out.shape)
        for name, layer in resnet18.named_children():
            if name == 'avgpool':
                out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
            elif name == 'fc':
                out = out.view(out.size(0), -1)
                out = layer(out)
                fc_list.append(out.flatten())

                softmax_outputs = F.softmax(out, dim=1)
                _, preds = torch.max(softmax_outputs, 1)
                binary_list.extend((preds == labels).cpu().numpy())
            else:
                out = layer(out)
                if name == 'layer1':
                    layer1_list.append(out.flatten())
                if name == 'layer2':
                    layer2_list.append(out.flatten())
                if name == 'layer3':
                    layer3_list.append(out.flatten())
                if name == 'layer4':
                    layer4_list.append(out.flatten())
            output_shapes[name] = out.shape
    
    predictor_layer1_dataset = PredictorDataset(layer1_list, fc_list)
    predictor_layer2_dataset = PredictorDataset(layer2_list, fc_list)
    predictor_layer3_dataset = PredictorDataset(layer3_list, fc_list)
    predictor_layer4_dataset = PredictorDataset(layer4_list, fc_list)
    selector_dataset = SelectorDataset(fc_list, binary_list)
    print(f"Time to build dataset: {time.time() - total_start}")
    # Create dataloaders
    batch_size = 32
    predictor_layer1_data_loader = DataLoader(predictor_layer1_dataset, batch_size=batch_size, shuffle=True)
    predictor_layer2_data_loader = DataLoader(predictor_layer2_dataset, batch_size=batch_size, shuffle=True)
    predictor_layer3_data_loader = DataLoader(predictor_layer3_dataset, batch_size=batch_size, shuffle=True)
    predictor_layer4_data_loader = DataLoader(predictor_layer4_dataset, batch_size=batch_size, shuffle=True)
    selector_data_loader = DataLoader(selector_dataset, batch_size=batch_size, shuffle=True)
    print(f"Time to build DataLoaders: {time.time() - total_start}")



    '''Train Predictors'''
    from sample import PredictorNetwork, SelectorNetwork
    from sample import BaseModel
    # Example training function for predictor and selector networks

    layer1_flat = layer1_list[0].flatten()  #flat tensor
    input_dim = layer1_flat.shape[0]  # This will be a tuple with a single element (total number of elements)
    output_dim = 320 #### flatten w.r.t. each batch?

    predictor_model = PredictorNetwork(input_dim, output_dim).to(device)
    selector_model = SelectorNetwork(output_dim).to(device)

    predictor_loss = train_predictor(predictor_model, predictor_layer1_data_loader)
    selector_loss = train_selector(selector_model, selector_data_loader)






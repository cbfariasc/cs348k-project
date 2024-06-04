from sample import BaseModel, train_predictor, train_selector
from sample import PredictorNetwork
from sample import SelectorNetwork
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import time
from tqdm import tqdm
import argparse
TQDM_DISABLE = False
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from dataset_utils import PredictorDataset, SelectorDataset

def train_predictor(model, dataloader, epochs=10):
    print("Training predictor")
    total_start = time.time()
    print("Start time: ", total_start)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    model.train()
    loss_history = []
    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", disable=TQDM_DISABLE):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds, Loss: {running_loss / len(dataloader):.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
    total_duration = time.time() - total_start
    print(f"Total training time: {total_duration:.2f} seconds")

    return loss_history

def train_selector(model, dataloader, epochs=10):
    print("Training selector")
    total_start = time.time()
    print("Start time: ", total_start)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    model.train()
    loss_history = []
    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", disable=TQDM_DISABLE):
            inputs, targets = inputs.to(device), targets.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds, Loss: {running_loss / len(dataloader):.4f}")

        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
    total_duration = time.time() - total_start
    print(f"Total training time: {total_duration:.2f} seconds")

    return loss_history

def train_models(train_model_type):
    """Set up the DataLoaders: """
    # Define the transformation for the validation data
    batch_size = 1
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
    print("Loading CIFAR10 dataset to base model")
    total_start = time.time()
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"total time: {time.time() - total_start}")


    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_size = len(full_train_dataset) #int(0.1 * len(full_train_dataset)) # 6400
    val_size = len(val_dataset) # len(full_train_dataset) - train_size

    #train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    test_size = int(0.1 * len(test_dataset))
    _, small_test_dataset = random_split(test_dataset, [len(test_dataset) - test_size, test_size])

    test_loader = DataLoader(small_test_dataset, batch_size=batch_size, shuffle=False)

    # Print the sizes of the datasets
    print(f'Training dataset size: {len(full_train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(small_test_dataset)}')


    '''Instantiate models, criterion, and optimizer'''
    # base_model = BaseModel()
    base_model = BaseModel()
    base_model.eval()
    num_classes = 10 # CIFAR10

    # Load the pretrained ResNet-50 model
    resnet50 = models.resnet50(pretrained=True) # pretrained on CIFAR10
    # resnet50.eval() # sets this to evaluation mode

    resnet18 = models.resnet18(pretrained=True)
    for param in resnet18.parameters():
        param.requires_grad = False
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes) #rewrites resnet18 final fc layer


    '''Build the predictor/selector Database'''


    output_shapes = {}

    resnet18 = resnet18.to(device)
    resnet18.eval()

    layer1_list, fc_list = [], []
    binary_list = []


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(len(val_dataset))

    num_samples = 8000  # Adjust this number as needed
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    val_subset = Subset(val_dataset, indices)
    val_subset_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

    if train_model_type == "pred":
        print("Collecting Predictor/selector data", train_size, "samples")
        total_start = time.time()
        comparison_results  = []
        with torch.no_grad():
            for images, labels in val_subset_loader: 
                images = images.to(device)
                out = images
                labels = labels.to(device)
                # print(out.shape)
                for name, layer in resnet18.named_children():
                    if name == 'avgpool':
                        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
                    elif name == 'fc':
                        out = out.view(out.size(0), -1)
                        out = layer(out)

                        out_tensor = torch.tensor(out)
                        out_final = out_tensor.reshape(out_tensor.shape[0], -1)  #flat tensor
                        fc_list.append(out_final)

                        softmax_outputs = F.softmax(out, dim=1)
                        _, preds = torch.max(softmax_outputs, 1)
                        comparison = preds == labels
                        comparison_results.append(comparison)
                    else:
                        out = layer(out)
                        out_tensor = torch.tensor(out)
                        out_temp = out_tensor.reshape(out_tensor.shape[0], -1)
                        if name == 'layer1':
                            layer1_list.append(out_temp)

                    output_shapes[name] = out.shape
        stacked_tensor = torch.stack(comparison_results)
        accuracy = torch.sum(stacked_tensor)
        print(f"base model accuracy: {accuracy}")

        final_results = torch.cat(comparison_results).tolist()
        binary_list.extend(final_results)
        predictor_layer1_dataset = PredictorDataset(layer1_list, fc_list)
        print(f"Time to build Predictor dataset: {time.time() - total_start}")
      
        predictor_layer1_data_loader = DataLoader(predictor_layer1_dataset, batch_size=batch_size, shuffle=True)
        print(f"Time to build Predictor DataLoader: {time.time() - total_start}")


  
        '''Train Predictors'''
        # Example training function for predictor and selector networks
        layer1_flat = layer1_list[0]  # (B, P, P)
        input_dim = layer1_flat.shape[1] # This will be a tuple with a single element (total number of elements)
        output_dim = 10 #### flatten w.r.t. each batch?
        # print(f"input dim= {input_dim}")
        # print(f"output dim= {output_dim}")

        predictor_model = PredictorNetwork(input_dim, output_dim).to(device)
        predictor_loss = train_predictor(predictor_model, predictor_layer1_data_loader)
        p_save_path = 'models/p_layer1_v1.pth'
        # Save the model's state dictionary
        torch.save(predictor_model.state_dict(), p_save_path)
    
    else:
        '''Train Selector Network'''
        # Train the selector: 
        # Run the predictor, get its outputs, get the true value and the predictor value. 
        # IN: Predictor output (predicting FC).....  OUT: 1  (if predictor_out == actual FC)
        cache_hit_list = []
        pred_out_list = []

        # Load in predictor 
        input_dim = 200704
        output_dim = 10
        predictor_model = PredictorNetwork(input_dim, output_dim).to(device)
        p_model_path = "models/p_layer1_v1.pth"
        predictor_model.load_state_dict(torch.load(p_model_path))
        predictor_model.eval()

        # Collect predictor and model outputs for selector dataset
        with torch.no_grad():
            for images, labels in val_subset_loader: 
                images = images.to(device)
                out = images
                labels = labels.to(device)
                # print(out.shape)
                for name, layer in resnet18.named_children():
                    if name == 'avgpool':
                        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
                    elif name == 'fc':
                        out = out.view(out.size(0), -1)
                        out = layer(out)

                        out_tensor = torch.tensor(out)
                        out_final = out_tensor.reshape(out_tensor.shape[0], -1)  #flat tensor
                        fc_list.append(out_final)

                        softmax_outputs = F.softmax(out, dim=1)
                        _, preds = torch.max(softmax_outputs, 1)

                        softmax_predictor = F.softmax(pred_out, dim=1)
                        _, predictor_label = torch.max(softmax_predictor, 1)

                        cache_hit = predictor_label == preds # checks the predictor's real accuracy of the model output
                        cache_hit_list.append(cache_hit)
                    else:
                        out = layer(out)
                        out_tensor = torch.tensor(out)
                        out_temp = out_tensor.reshape(out_tensor.shape[0], -1)
                        if name == 'layer1':
                            pred_out = predictor_model(out_temp)
                            pred_out_list.append(pred_out)
                    output_shapes[name] = out.shape

        # Train selector
        selector_dataset = SelectorDataset(pred_out_list, cache_hit_list)
        print("sel data len")
        print(len(selector_dataset))
        selector_data_loader = DataLoader(selector_dataset, batch_size=batch_size, shuffle=True)
        selector_model = SelectorNetwork(output_dim).to(device)
        selector_loss = train_selector(selector_model, selector_data_loader)
        s_save_path = 'models/s_layer1_v1.pth'
        torch.save(selector_model.state_dict(), s_save_path)


def test_models():
    batch_size = 1 #test with batch size of 1-2 to allow you to skip layers 
    input_dim = 200704
    output_dim = 10

    resnet = models.resnet18(pretrained=True).to(device)
    selector = SelectorNetwork(output_dim).to(device)
    predictor = PredictorNetwork(input_dim, output_dim).to(device)
    resnet.fc = nn.Linear(resnet.fc.in_features, 10).to(device)

    p_model_path = "models/p_layer1_v1.pth"
    s_model_path = "models/s_layer1_v1.pth"

    selector.load_state_dict(torch.load(s_model_path))
    predictor.load_state_dict(torch.load(p_model_path))

    resnet.eval()
    selector.eval()
    predictor.eval()
    # Transformation and data loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Iterate through each layer in the ResNet-50 model and apply them sequentially
    total_start = time.time()
    
    num_sample_fc = 0
    num_sample_layer1 = 0
    num_correct_fc = 0
    num_correct_layer1 = 0
    total_samps = 0
    with torch.no_grad():
        for image, labels in val_loader:
          #image = image.to(device)
          output = image.to(device)
          labels = labels.to(device)
          total_samps += 1
          
          for name, layer in resnet.named_children():
              #start = time.time()
              if name == 'avgpool':
                  output = nn.functional.adaptive_avg_pool2d(output, (1, 1))

              elif name == 'fc':
                  output = output.view(output.size(0), -1)
                  output = layer(output.to(device))
                  softmax_outputs = F.softmax(output, dim=1)
                  _, preds = torch.max(softmax_outputs, 1)
                  num_sample_fc += 1
                  if preds == labels:
                      num_correct_fc += (preds == labels).sum().item()

              elif name == 'layer1':
                  output = layer(output)
                  pred_out = predictor(output.to(device))
                  if selector(pred_out) == 1:
                      print("cache hit!")
                      num_sample_layer1 += 1
                      output = pred_out
                      if output == labels:
                          num_correct_layer1 += (output == labels).sum().item()
                      else:
                          print("incorrect cache hit")
                          print(output)
                          print(labels)
                      break    
                  
              else:
                  output = layer(output)


    print(f"total accuracy for fc layer: {num_correct_fc / num_sample_fc}")
    if num_sample_layer1 != 0:
        print(f"total accuracy for layer1: {num_correct_layer1 / num_sample_layer1}")
    print(f"percent cache hit: {num_sample_layer1 / total_samps}")
    print(f"total time: {time.time() - total_start}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--train_model_type", type=str, default="pred")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train_models(args.train_model_type)
    else:
        test_models()




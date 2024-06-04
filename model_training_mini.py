from sample import BaseModel, train_predictor, train_selector
from sample import PredictorNetwork
from sample import SelectorNetwork
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
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np
from dataset_utils import PredictorDataset, SelectorDataset
import random

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

def sample_tensors(outputs, labels, k):
    # Step 1: Identify the indices of elements in labels that are True and False
    indices_true = [i for i, label in enumerate(labels) if label]
    indices_false = [i for i, label in enumerate(labels) if not label]
    
    # Check if there are enough elements with True and False labels
    if len(indices_true) < k:
        raise ValueError(f"There are only {len(indices_true)} elements with label True, but {k} were requested.")
    if len(indices_false) < k:
        raise ValueError(f"There are only {len(indices_false)} elements with label False, but {k} were requested.")
    
    # Step 2: Randomly select k of these indices from each group
    selected_indices_true = random.sample(indices_true, k)
    selected_indices_false = random.sample(indices_false, k)
    
    # Step 3: Create the truncated lists for each group
    truncated_outputs_true = [torch.tensor(outputs[i]) for i in selected_indices_true]
    truncated_labels_true = [torch.tensor(labels[i]) for i in selected_indices_true]
    
    truncated_outputs_false = [torch.tensor(outputs[i]) for i in selected_indices_false]
    truncated_labels_false = [torch.tensor(labels[i]) for i in selected_indices_false]

    outs = []
    trunc_labs = []
    for i in range(k):
        outs.append(truncated_outputs_true[i])
        outs.append(truncated_outputs_false[i])
        trunc_labs.append(truncated_labels_true[i])
        trunc_labs.append(truncated_labels_false[i])
    
    return outs, trunc_labs

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
            loss = criterion(outputs.squeeze(dim=0), targets)
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

    # Load the validation dataset
    print("Loading RESNET-mini dataset to base model")
    total_start = time.time()
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dir = 'data/imagenet-mini/train'
    val_dir = 'data/imagenet-mini/val'

    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
    # print('datset sizes: ')
    # print(len(train_dataset))
    # print(len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"base Data Loaded! total time: {time.time() - total_start}")

    train_size = len(train_loader)
    val_size = len(val_loader) 

    '''Instantiate models, criterion, and optimizer'''
    base_model = BaseModel()
    base_model.eval()
    num_classes = 1000 # ImageNet

    resnet18 = models.resnet18(pretrained=True)
    for param in resnet18.parameters():
        param.requires_grad = False


    '''Build the predictor/selector Database'''
    output_shapes = {}
    resnet18 = resnet18.to(device)
    resnet18.eval()

    layer1_list, fc_list = [], []
    binary_list = []

    print(len(val_dataset))
    # num_samples = min(10)
    # indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    # val_subset = Subset(val_dataset, indices)
    # val_subset_loader = DataLoader(val_subset, batch_size=1, shuffle=False)


    if train_model_type == "pred":
        print("Collecting Predictor/selector data", train_size, "samples")
        total_start = time.time()
        comparison_results  = []
        with torch.no_grad():
            for images, labels in val_loader:   #val or subset
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
        output_dim = 1000 #### flatten w.r.t. each batch?
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
        output_dim = 1000
        predictor_model = PredictorNetwork(input_dim, output_dim).to(device)
        p_model_path = "models/p_layer1_v1.pth"
        predictor_model.load_state_dict(torch.load(p_model_path))
        predictor_model.eval()
        num_data_cache_hit = 0

        # Collect predictor and model outputs for selector dataset
        with torch.no_grad():
            for images, labels in val_loader: 
                out = images.to(device)
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
                        # fc_list.append(out_final)

                        softmax_outputs = F.softmax(out, dim=1)
                        _, preds = torch.max(softmax_outputs, 1)

                        softmax_predictor = F.softmax(pred_out, dim=1)
                        _, predictor_label = torch.max(softmax_predictor, 1)

                        cache_hit = predictor_label == preds # checks the predictor's real accuracy of the model output
                        # cache_hit_final = cache_hit.reshape(cache_hit.shape[0], -1)
                        # print(cache_hit.shape)
                        if cache_hit:
                            num_data_cache_hit += 1
                        cache_hit_list.append(cache_hit)
                    else:
                        out = layer(out)
                        out_tensor = torch.tensor(out)
                        out_temp = out_tensor.reshape(out_tensor.shape[0], -1)
                        if name == 'layer1':
                            pred_out = predictor_model(out_temp)
                            #print(f"pred out = {pred_out.shape}")
                            pred_out_temp = torch.tensor(pred_out)
                            pred_out_final = pred_out_temp.reshape(pred_out_temp.shape[0], -1)
                            pred_out_list.append(pred_out_final)
                    output_shapes[name] = out.shape
        # Train selector
        print(f"number cache hits in data {num_data_cache_hit}")
        binary_list = []
        final_results = torch.cat(cache_hit_list).tolist()
        binary_list.extend(final_results)

        pred_out_trunc, cache_hit_trunc = sample_tensors(pred_out_list, cache_hit_list, num_data_cache_hit)

        # print("pred_out_trunc")
        # print(type(pred_out_trunc[0]))
        # print("cache_hit_trunc")
        # print(type(cache_hit_trunc[0]))
        
        selector_dataset = SelectorDataset(pred_out_trunc, cache_hit_trunc)
        print("sel data ")
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
                  # print(f'layer 1 pred out {selector(pred_out)}')
                  if selector(pred_out) > 0.55:
                      #print("cache hit!")
                      num_sample_layer1 += 1
                      output = pred_out
                      softmax_outputs = F.softmax(output, dim=1)
                      _, preds = torch.max(softmax_outputs, 1)
                      #if output == labels:
                      print("preds")
                      print(preds)
                      print("labels")
                      print(labels)
                      num_correct_layer1 += (preds == labels).sum().item()
                      #else:
                      #    print("incorrect cache hit")
                      #    print(output)
                      #    print(labels)
                      break    
                  
              else:
                  output = layer(output)


    # print(f"total accuracy for fc layer: {num_correct_fc / num_sample_fc}") tests model accuracy when it doesn't cache hit only
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




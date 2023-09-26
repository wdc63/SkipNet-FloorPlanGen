import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

from utils import load_pickle
from metrics import mIOU
from models import GraphFloorplanUNet
from data import msdDataset

num_node_features = 4
classes = 11

"""
'Bedroom': 0,
'Livingroom': 1,
'Kitchen': 2,
'Dining': 3, 
'Corridor': 4,
'Stairs': 5,
'Storeroom': 6,
'Bathroom': 7,
'Balcony': 8,
'Structure': 9,
'Background': 13 -> 10
"""

# Specify the device to use
device = "cuda:0"

def train_model(model, train_loader, optimizer, criterion, classes, epochs):
    model.train()
    
    if not os.path.exists('results'):
        os.makedirs('results')
    current_time = datetime.now().strftime('%m%d%H%M')
    run_dir = os.path.join('results', current_time)
    os.makedirs(run_dir)
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_iou = 0
        # A progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for images, graph_data, gt_images in progress_bar: 
            images = images.to(device)
            graph_data = graph_data.to(device)
            gt_images = gt_images.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, graph_data)
            
            weight = torch.ones(classes)
            weight[-2] = 2
            weight[-1] = 2
            weight = weight.to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=weight)
            
            # Adjust class values
            gt_images[gt_images == 13] = 10
            loss = criterion(outputs, gt_images)

            # Compute IoU
            pred_tensor = torch.argmax(outputs, dim=1)
            iou = mIOU(pred_tensor, gt_images, classes)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_iou += iou

            # Update the progress bar
            progress_bar.set_postfix({'loss': loss.item(), 'mIoU': iou})

        epoch_loss = epoch_loss / len(train_loader)
        epoch_iou = epoch_iou / len(train_loader)
        
        print(f'Epoch {epoch}, Average Loss: {epoch_loss}, Average mIoU: {epoch_iou}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for images, graph_data, gt_images in val_loader:
                images = images.to(device)
                graph_data = graph_data.to(device)
                gt_images = gt_images.to(device)
                
                # Forward pass
                outputs = model(images, graph_data)
                # Adjust class values
                gt_images[gt_images == 13] = 10
                loss = criterion(outputs, gt_images)

                # Compute IoU
                pred_tensor = torch.argmax(outputs, dim=1)
                iou = mIOU(pred_tensor, gt_images, classes)

                val_loss += loss.item()
                val_iou += iou

        val_loss = val_loss / len(val_loader)
        val_iou = val_iou / len(val_loader)
        
        print(f'Epoch {epoch}, Average Validation Loss: {val_loss}, Average Validation mIoU: {val_iou}')
        
        # Save model every 5 epochs
        if epoch % 10 == 0:
            save_path = os.path.join(run_dir, f'model_checkpoint_epoch_{epoch}.pt')
            torch.save(model.state_dict(), save_path)

            
input_nc = 3
output_nc = 11

# Create the model
model = GraphFloorplanUNet(num_node_features, input_nc, output_nc, features=[64, 128, 256, 512]).to(device)

# Choose a criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# DataLoader
train_dataset = msdDataset('/data/iccvw/CHALLENGE/dataset_struct_in_aug/train')
val_dataset = msdDataset('/data/iccvw/CHALLENGE/dataset_struct_in_aug/val')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Train the model
train_model(model, train_loader, optimizer, criterion, classes, epochs=500)
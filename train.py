import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from F7053T_Pressure_Identification.dataset import dataset
from F7053T_Pressure_Identification.model import pressureInsolesTransformer


path_list = ['data/subject1.pth', 'data/subject2.pth', 'data/subject3.pth', 'data/subject4.pth', 'data/subject5.pth',
                        'data/subject6.pth', 'data/subject7.pth', 'data/subject8.pth', 'data/subject9.pth', 'data/subject10.pth']
data = dataset(path_list)

# parameters:
batch_size = 10
train_set, validation_set, test_set = random_split(data, [0.6, 0.2, 0.2])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = []

    for data, label in iter(loader):
        data = data.permute([0, 2, 1]).to(torch.float32).to(device)
        label = label.permute([0, 2, 1]).to(torch.float32).to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        
    return running_loss


def validate(model, loader, criterion, device, best_loss=float('inf'), save_path="best_model.pth"):
    model.eval()
    running_loss = []

    with torch.no_grad():
        for data, label in iter(loader):
            data = data.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)
            outputs = model(data)
            loss = criterion(outputs, label)
            running_loss.append(loss.item())
    val_loss = sum(running_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), save_path)

    return running_loss, best_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        val_loss, best_loss = validate(model, val_loader, criterion, device, best_loss, save_path="best_autoencoder.pth")
        val_losses.append(val_loss)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {sum(train_loss):.4f}"
              f"| Val Loss: {sum(val_loss):.4f}", end="\r")
    return train_losses

def reinit_transformer_weights(model, d_model):
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)

            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)

        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1/math.sqrt(d_model))

if __name__ == "__main__":
    device = torch.device("mps")
    epochs = 200
    
    model = pressureInsolesTransformer(
    input_dim=302,
    latent_dim=32,
    num_classes=10,
    num_encoder_layers=6,
    nhead=16,
    dim_feedforward=1024,
    dropout=0.3,
    seq_len=339
    )
    reinit_transformer_weights(model, 1024)
    model_path = "./best_autoencoder.pth"
    model.to(device)   
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.2)
    scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=20)
    scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], [21])
    train_losses, val_losses = train_model(model, train_loader, validation_loader, criterion, optimizer, scheduler, epochs, device)



import torch
import torch.nn as nn
import torch.optim as optim
from model.spam_detector import SpamDetector
from model.data_loader import load_data
from utils.class_weights import get_class_weights
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, X_test, y_train, y_test, vocab_size = load_data("data/spam.csv")

X_train = torch.tensor(X_train).long().to(device)
y_train = torch.tensor(y_train).float().to(device)

model = SpamDetector(vocab_size).to(device)

weights = get_class_weights(y_train.cpu().numpy())
pos_weight = torch.tensor(3.0).to(device)


criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_loss = float('inf')
patience = 3
counter = 0

for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), "best_model.pt")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

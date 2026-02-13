
import torch
import numpy as np
from model.spam_detector import SpamDetector
from model.data_loader import load_data
from utils.metrics import print_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, X_test, y_train, y_test, vocab_size = load_data("data/spam.csv")

X_test = torch.tensor(X_test).long().to(device)

model = SpamDetector(vocab_size).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

with torch.no_grad():
    outputs = model(X_test).squeeze()
    preds = torch.sigmoid(outputs).cpu().numpy()
    preds = (preds > 0.4).astype(int)


print_metrics(y_test, preds)

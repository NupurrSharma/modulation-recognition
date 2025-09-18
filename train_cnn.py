import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from signal_utils import make_spectrogram

class CNNModClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNModClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32*4*4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_dataset(source="synthetic"):
    if source=="synthetic":
        X = np.load("dataset/signals.npy")
        y = np.load("dataset/labels.npy")
        X = np.array([make_spectrogram(sig) for sig in X])
        return X, y, list(range(len(np.unique(y))))
    elif source=="radioml":
        from radioml_loader import load_radioml_dataset
        X, y, class_names = load_radioml_dataset("RML2016.10a_dict.pkl",
                                                 snr_list=[0,10,20],
                                                 max_samples_per_class=1000)
        X = np.array([make_spectrogram(sig) for sig in X])
        return X, y, class_names
    else:
        raise ValueError("Unknown dataset source")

def train_model(X, y, class_names, epochs=30, batch_size=32, lr=0.001, save_path="cnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X / np.max(X)
    X = X[:, np.newaxis, :, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train,dtype=torch.float32),
                                            torch.tensor(y_train,dtype=torch.long)),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test,dtype=torch.float32),
                                           torch.tensor(y_test,dtype=torch.long)),
                             batch_size=batch_size)
    model = CNNModClassifier(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true,y_pred)*100
    print(f"🎯 CNN Accuracy: {acc:.2f}%")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved as {save_path}")

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic","radioml"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    X, y, class_names = load_dataset(args.dataset)
    save_path = f"cnn_classifier_{args.dataset}.pth"
    train_model(X, y, class_names, epochs=args.epochs, batch_size=args.batch_size, save_path=save_path)

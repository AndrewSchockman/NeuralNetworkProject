# Import dependencies
import time
import torch
from torch import nn, save
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---- Data (normalize + shuffle) ----
# MNIST normalization constants
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

# Train + Val datasets/loaders
train = datasets.MNIST(root="data", download=True, train=True,  transform=transform)
val   = datasets.MNIST(root="data", download=True, train=False, transform=transform)

# keep your old name "dataset" for the training loader
dataset    = DataLoader(train, batch_size=64, shuffle=True)
val_loader = DataLoader(val,   batch_size=256, shuffle=False)

# ---- Image Classifier Neural Network ----
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # keep attribute name "model" so your test file can import this class and load weights
        self.model = nn.Sequential(
            # 28x28 -> 28x28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 14 -> 7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # -> (N,128,1,1)
            nn.Flatten(),             # -> (N,128)
            nn.Dropout(0.2),
            nn.Linear(128, 10)        # 10 classes for digits 0–9
        )

    def forward(self, x):
        return self.model(x)

# Instance of the neural network, loss, optimizer
device = "cpu"  # CAN CHANGE TO CUDA IF AVAILABLE
clf = ImageClassifier().to(device)
opt = Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)

# ---- Training flow ----
if __name__ == "__main__":
    torch.manual_seed(0)

    epochs = 12
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        t0 = time.perf_counter()
        clf.train()
        running_loss = 0.0

        for X, y in dataset:
            X, y = X.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = clf(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * X.size(0)

        train_loss = running_loss / len(train)

        # ---- validation ----
        clf.eval()
        val_running = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = clf(X)
                val_running += loss_fn(logits, y).item() * X.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss = val_running / len(val)
        val_acc = correct / total
        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            # store a CPU copy so it loads anywhere
            best_state = {k: v.detach().cpu() for k, v in clf.state_dict().items()}

        dt = time.perf_counter() - t0
        print(f"Epoch:{epoch} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4%} | {dt:.1f}s")

    # Save the best model to model_state.pt (same filename you used before)
    if best_state is not None:
        clf.load_state_dict(best_state)
    with open('model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)
    print(f"Saved best model (val_acc={best_acc:.4%}) to model_state.pt")

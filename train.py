import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# ekran karti varsa onu kullan yoksa cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanilan cihaz:", device)

# hiperparametreler (raporda yazilacak)
batch_size = 64
learning_rate = 0.001
epochs = 10

# mnist goruntulerini tensore cevirip normalize ediyoruz
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # mnist standart degerleri
])

# veri setini indiriyoruz
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 60 binlik egitim verisini 50 bin train, 10 bin validation olarak boluyoruz
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

# Temel MLP Modeli (CNN yok, sadece Linear katmanlar)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 28x28 = 784 piksel giris
        self.fc1 = nn.Linear(28 * 28, 256) # 1. hidden layer
        self.fc2 = nn.Linear(256, 128)     # 2. hidden layer
        self.fc3 = nn.Linear(128, 10)      # cikis katmani (0-9 arasi 10 sinif)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28) # resmi duzlestirme (flatten)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) # crossentropy kullandigimiz icin burda softmax'e gerek yok
        return x

model = MLP().to(device)

# loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# grafik cizdirmek icin listeler
train_losses = []
val_losses = []
train_accs = []
val_accs = []

print("\nEgitim basliyor...")
baslangic_zamani = time.time()

for epoch in range(epochs):
    # --- TRAIN KISMI ---
    model.train()
    running_loss = 0.0
    dogru = 0
    toplam = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        toplam += labels.size(0)
        dogru += (predicted == labels).sum().item()
        
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = 100 * dogru / toplam
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_dogru = 0
    val_toplam = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_toplam += labels.size(0)
            val_dogru += (predicted == labels).sum().item()
            
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = 100 * val_dogru / val_toplam
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)
    
    print(f"Epoch [{epoch+1}/{epochs}] -> "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: %{epoch_train_acc:.2f} | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: %{epoch_val_acc:.2f}")

bitis_zamani = time.time()
gecen_sure = bitis_zamani - baslangic_zamani
print(f"\nEgitim bitti! Toplam sure: {gecen_sure:.2f} saniye")

# Test
model.eval()
test_dogru = 0
test_toplam = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_toplam += labels.size(0)
        test_dogru += (predicted == labels).sum().item()

test_acc = 100 * test_dogru / test_toplam
print(f"Test Accuracy: %{test_acc:.2f}")

# modeli kaydetme
torch.save(model.state_dict(), "mlp_model.pth")
print("Model 'mlp_model.pth' olarak kaydedildi.")

# Grafik cizdirme
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 5))

# Loss grafigi
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss', marker='o')
plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o')
plt.title('Training ve Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy grafigi
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label='Training Accuracy', marker='o')
plt.plot(epochs_range, val_accs, label='Validation Accuracy', marker='o')
plt.title('Training ve Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('egitim_grafikleri.png')
print("Grafikler 'egitim_grafikleri.png' olarak kaydedildi.")
plt.show()

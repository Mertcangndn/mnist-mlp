import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()
model.load_state_dict(torch.load("mlp_model.pth", map_location="cpu"))
model.eval()

klasor = "numbers"
dosyalar = sorted(os.listdir(klasor))

tahminler = []
gorseller = []

for dosya in dosyalar:
    if not dosya.endswith((".png", ".jpg", ".jpeg")):
        continue
        
    yol = os.path.join(klasor, dosya)
    img = cv2.imread(yol, cv2.IMREAD_GRAYSCALE)
    
    # arka plan beyazsa ters cevir
    if img.mean() > 127:
        img = cv2.bitwise_not(img)
        
    # silik cizgileri tam beyaz yap ve kalinlastir (28x28'e kuculunce kaybolmasin diye)
    _, img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    
    # rakamin etrafindaki fazla bosluklari kirpip 20x20'ye sigdir, sonra 28x28 ortala (mnist formati)
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        crop = img[y:y+h, x:x+w]
        
        # oran bozulmasin diye kare icine al
        max_dim = max(w, h)
        kare = np.zeros((max_dim, max_dim), dtype=np.uint8)
        y_off = (max_dim - h) // 2
        x_off = (max_dim - w) // 2
        kare[y_off:y_off+h, x_off:x_off+w] = crop
        
        img_20 = cv2.resize(kare, (20, 20), interpolation=cv2.INTER_AREA)
        img_resized = np.pad(img_20, ((4, 4), (4, 4)), mode='constant', constant_values=0)
    else:
        img_resized = cv2.resize(img, (28, 28))

    gorseller.append(img_resized)
    
    tensor_img = torch.tensor(img_resized, dtype=torch.float32) / 255.0
    tensor_img = (tensor_img - 0.1307) / 0.3081
    
    with torch.no_grad():
        cikti = model(tensor_img)
        tahmin = torch.argmax(cikti, dim=1).item()
        tahminler.append(tahmin)
        print(f"{dosya} -> Tahmin: {tahmin}")

print("\nOkul Numarasi Tahmini:", "".join(map(str, tahminler)))

plt.figure(figsize=(12, 3))
for i in range(len(gorseller)):
    plt.subplot(1, len(gorseller), i + 1)
    plt.imshow(gorseller[i], cmap="gray")
    plt.title(f"Tahmin: {tahminler[i]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("numara_sonuc.png")
plt.show()

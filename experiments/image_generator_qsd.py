import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# Настройки GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство для генерации: {device}")

# 1. Функция генерации математических таргетов (картинок 64x64)
def generate_target_images(Y, E):
    num_samples = len(Y)
    images = np.zeros((num_samples, 1, 64, 64))
    
    x = np.linspace(-3, 3, 64)
    y = np.linspace(-3, 3, 64)
    X_grid, Y_grid = np.meshgrid(x, y)
    r = np.sqrt(X_grid**2 + Y_grid**2)
    
    for i in range(num_samples):
        class_idx = Y[i]
        entropy = E[i]
        
        if class_idx == 0: # Круги
            img = np.sin(r * 4.0)
        elif class_idx == 1: # Чистая интерференция (зебра), частота от энтропии
            freq = 2.0 + entropy * 8.0
            img = np.sin(X_grid * freq + Y_grid * freq)
        elif class_idx == 2: # Искаженная квантовым шумом зебра
            freq = 2.0 + entropy * 8.0
            noise = np.sin(X_grid * 15.0) * 0.4
            img = np.sin(X_grid * freq + Y_grid * freq + noise)
        else: # Класс 3 - Полный тепловой хаос
            img = np.random.normal(0, 0.5, size=(64, 64))
            
        # Нормализуем в диапазон [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        images[i, 0, :, :] = img
        
    return images

# Загружаем данные из Шага 1
X_mats = np.load("data/X_matrices.npy")
Y_labs = np.load("data/Y_labels.npy")
E_ents = np.load("data/E_entropies.npy")

# Генерируем картинки-таргеты
print("Рендеринг целевых математических изображений...")
X_imgs = generate_target_images(Y_labs, E_ents)

# Переводим в тензоры
X_mats_t = torch.tensor(X_mats, dtype=torch.float32)
X_imgs_t = torch.tensor(X_imgs, dtype=torch.float32)

dataset = TensorDataset(X_mats_t, X_imgs_t)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Архитектура Полного Пайплайна QSD (Энкодер-Адаптер-Декодер)
class QSDFullGenerator(nn.Module):
    def __init__(self):
        super(QSDFullGenerator, self).__init__()
        
        # Квантовый суррогатный энкодер (из прошлого шага)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=2, stride=2), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2), # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        # Наш Линейный проектор-адаптер (разворачивает латентный вектор в карту признаков)
        self.latent_projector = nn.Linear(128, 64 * 4 * 4)
        
        # Декодер (аналог генеративного пути Stable Diffusion, разворачивает 4x4 в 64x64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),    # 32x32 -> 64x64
            nn.Sigmoid() # Выход в диапазоне [0, 1]
        )

    def forward(self, rho):
        latent = self.encoder(rho)
        projected = self.latent_projector(latent).view(-1, 64, 4, 4)
        generated_img = self.decoder(projected)
        return generated_img

model = QSDFullGenerator().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Цикл обучения генерации
epochs = 40
print("Старт генеративного обучения...")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_rho, batch_img in train_loader:
        batch_rho, batch_img = batch_rho.to(device), batch_img.to(device)
        
        optimizer.zero_grad()
        output_img = model(batch_rho)
        loss = criterion(output_img, batch_img)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_rho.size(0)
        
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Эпоха [{epoch+1}/{epochs}] | Ошибка рендеринга (MSE): {epoch_loss / len(dataset):.5f}")

# 4. Визуализация и сохранение результатов проверки
print("\nГенерация финальных 'Паспортов декогеренции'...")
model.eval()

# Выберем по одному знаковому примеру каждого класса напрямую из датасета
indices = [0, 250, 500, 750] # Первые элементы классов 0, 1, 2, 3
class_names = ["Сепарабельное (Класс 0)", "Запутанное (Класс 1)", "Квантовый шум (Класс 2)", "Тепловой хаос (Класс 3)"]

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

with torch.no_grad():
    for idx, sample_idx in enumerate(indices):
        rho_input = X_mats_t[sample_idx].unsqueeze(0).to(device)
        pred_img = model(rho_input).cpu().squeeze().numpy()
        true_img = X_imgs[sample_idx].squeeze()
        
        # Строка 1: Что должна была нарисовать идеальная математика
        axes[0, idx].imshow(true_img, cmap='gray')
        axes[0, idx].set_title(f"Таргет\n{class_names[idx]}", fontsize=9)
        axes[0, idx].axis('off')
        
        # Строка 2: Что СГЕНЕРИРОВАЛА нейросеть по матрице плотности
        axes[1, idx].imshow(pred_img, cmap='gray')
        axes[1, idx].set_title(f"QSD Выход\n(Из матрицы $\\rho$)", fontsize=9)
        axes[1, idx].axis('off')

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/qsd_validation_matrix.png")
print("[УСПЕХ] График сравнения сохранен в 'results/qsd_validation_matrix.png'!")

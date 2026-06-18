import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

# Фиксируем сид для воспроизводимости
np.random.seed(42)
torch.manual_seed(42)

# Настройки GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Функция преобразования матрицы в 4-канальный квантовый тензор
def matrix_to_4channel(rho):
    size = rho.shape[0]
    
    # Извлекаем компоненты
    diag_mask = np.eye(size, dtype=bool)
    off_diag_mask = ~diag_mask
    
    diag_re = np.zeros_like(rho, dtype=float)
    diag_re[diag_mask] = np.real(rho[diag_mask])
    
    off_diag_re = np.zeros_like(rho, dtype=float)
    off_diag_re[off_diag_mask] = np.real(rho[off_diag_mask])
    
    off_diag_im = np.zeros_like(rho, dtype=float)
    off_diag_im[off_diag_mask] = np.imag(rho[off_diag_mask])
    
    # Матрица индексов квантовых расстояний (структурная маска)
    structural_mask = np.fromfunction(lambda i, j: np.abs(i - j) / float(size), (size, size), dtype=float)
    
    # Собираем 4 канала: (4, 16, 16)
    return np.stack([diag_re, off_diag_re, off_diag_im, structural_mask], axis=0)

# 2. Новая архитектура суррогата с поддержкой 4 физических каналов
class QSD4ChannelNet(nn.Module):
    def __init__(self):
        super(QSD4ChannelNet, self).__init__()
        # Входных каналов теперь 4 вместо 2
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=2, stride=2), # 16x16 -> 8x8
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=2, stride=2), # 8x8 -> 4x4
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.LeakyReLU(0.1)
        )
        self.latent_projector = nn.Linear(128, 64 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),    # 32x32 -> 64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        projected = self.latent_projector(latent).view(-1, 64, 4, 4)
        return self.decoder(projected)

# 3. Подготовка синтетических данных для обучения новой архитектуры
print("Подготовка 4-канального датасета...")
X_mats = np.load("data/X_matrices.npy")
# Наш старый X_mats имеет форму (1000, 2, 16, 16) - восстановим из Re и Im исходные комплексные матрицы
X_complex = X_mats[:, 0, :, :] + 1j * X_mats[:, 1, :, :]

# Преобразуем весь датасет в 4 канала
X_4ch = np.array([matrix_to_4channel(rho) for rho in X_complex])

# Восстанавливаем таргеты изображений 64x64
Y_labs = np.load("data/Y_labels.npy")
E_ents = np.load("data/E_entropies.npy")

def generate_target_images_mini(Y, E):
    images = np.zeros((len(Y), 1, 64, 64))
    x, y = np.linspace(-3, 3, 64), np.linspace(-3, 3, 64)
    X_g, Y_g = np.meshgrid(x, y)
    r = np.sqrt(X_g**2 + Y_g**2)
    for i in range(len(Y)):
        if Y[i] == 0: img = np.sin(r * 4.0)
        elif Y[i] == 1: img = np.sin(X_g * (2.0 + E[i] * 8.0) + Y_g * (2.0 + E[i] * 8.0))
        elif Y[i] == 2: img = np.sin(X_g * (2.0 + E[i] * 8.0) + Y_g * (2.0 + E[i] * 8.0) + np.sin(X_g * 15.0) * 0.4)
        else: img = np.random.normal(0, 0.5, size=(64, 64))
        images[i, 0, :, :] = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return images

X_imgs = generate_target_images_mini(Y_labs, E_ents)

# 4. Экспресс-обучение новой модели
model = QSD4ChannelNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

inputs = torch.tensor(X_4ch, dtype=torch.float32).to(device)
targets = torch.tensor(X_imgs, dtype=torch.float32).to(device)

print("Обучение 4-канальной архитектуры (40 эпох)...")
model.train()
for epoch in range(40):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Эпоха [{epoch+1}/40] | Loss MSE: {loss.item():.5f}")

# 5. ГЕНЕРАЦИЯ ТЕСТОВОГО СОСТОЯНИЯ И ЕГО РАЗРУШЕНИЕ
print("\nЗапуск контролируемого разрушения на новой архитектуре...")

# Исходное состояние GHZ
phi = np.pi / 4.0
psi = np.zeros(16, dtype=complex)
psi[0] = 1.0 / np.sqrt(2)
psi[15] = np.exp(1j * phi) / np.sqrt(2)
rho_orig = np.outer(psi, np.conj(psi))

# Дефазировка (уничтожаем недиагональные элементы)
rho_deph = np.diag(np.diag(rho_orig))

# Рандомизация статистики (перемешиваем ненулевую классическую диагональ)
diag_elements = np.diag(rho_orig).copy()
np.random.shuffle(diag_elements)
rho_rand = np.diag(diag_elements)

# Переводим в 4-канальные тензоры
t_orig = torch.tensor(matrix_to_4channel(rho_orig), dtype=torch.float32).unsqueeze(0).to(device)
t_deph = torch.tensor(matrix_to_4channel(rho_deph), dtype=torch.float32).unsqueeze(0).to(device)
t_rand = torch.tensor(matrix_to_4channel(rho_rand), dtype=torch.float32).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    out_orig = model(t_orig).cpu().squeeze().numpy()
    out_deph = model(t_deph).cpu().squeeze().numpy()
    out_rand = model(t_rand).cpu().squeeze().numpy()

# Визуализация новой матрицы валидации
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(out_orig, cmap='gray')
axes[0].set_title("1. ρ_original\n(Квантовая структура)")
axes[0].axis('off')

axes[1].imshow(out_deph, cmap='gray')
axes[1].set_title("2. ρ_dephased\n(Только классическая диагональ)")
axes[1].axis('off')

axes[2].imshow(out_rand, cmap='gray')
axes[2].set_title("3. ρ_randomized\n(Разрушенная статистика)")
axes[2].axis('off')

plt.tight_layout()
output_path = "results/controlled_destruction_test_v2.png"
plt.savefig(output_path)
print(f"[УСПЕХ] График сохранен в '{output_path}'")

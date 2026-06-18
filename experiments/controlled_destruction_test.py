import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

# 1. Определение архитектуры суррогата (должна точно соответствовать обученной модели)
class QSDFullGenerator(nn.Module):
    def __init__(self):
        super(QSDFullGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.latent_projector = nn.Linear(128, 64 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, rho):
        latent = self.encoder(rho)
        projected = self.latent_projector(latent).view(-1, 64, 4, 4)
        generated_img = self.decoder(projected)
        return generated_img

# Настройки GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Математическая генерация трех состояний на базе ОДНОГО исходного
np.random.seed(101) # Фиксируем сид для воспроизводимости этого теста

# Генерируем базовое максимально запутанное состояние |GHZ> с фазовым сдвигом
phi = np.pi / 4.0
psi = np.zeros(16, dtype=complex)
psi[0] = 1.0 / np.sqrt(2)
psi[15] = np.exp(1j * phi) / np.sqrt(2)
rho_orig = np.outer(psi, np.conj(psi))

# Модификация 1: Полная дефазировка (уничтожение когерентности, сохранение вероятностей)
rho_deph = np.diag(np.diag(rho_orig)) # Оставляем только диагональ, остальное 0

# Модификация 2: Рандомизация статистики (перемешивание диагонали)
diag_elements = np.diag(rho_orig).copy()
np.random.shuffle(diag_elements)
rho_rand = np.diag(diag_elements)

# Формируем двухканальные вещественные тензоры (2, 16, 16) для нейросети
def to_tensor(rho):
    split = np.stack([np.real(rho), np.imag(rho)], axis=0)
    return torch.tensor(split, dtype=torch.float32).unsqueeze(0).to(device)

t_orig = to_tensor(rho_orig)
t_deph = to_tensor(rho_deph)
t_rand = to_tensor(rho_rand)

# 3. Инициализация модели и загрузка весов
# В реальном коде мы создаем новый инстанс. Так как веса в памяти прошлого скрипта, 
# мы сымитируем проверку на обученной модели, пересобрав полный пайплайн обучения 
# или просто инициализировав модель (для локального теста мы можем переобучить ее за секунды)

print("Инициализация пайплайна и проведение теста контролируемого разрушения...")
model = QSDFullGenerator().to(device)

# Для того чтобы тест прошел на обученной модели, подгрузим веса, если вы сохранили модель, 
# либо инициализируем генерацию. Чтобы не зависеть от файлов, обучим модель быстро заново на лету на 10 эпохах:
X_mats = np.load("data/X_matrices.npy")
# Восстановим таргеты
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

Y_labs = np.load("data/Y_labels.npy")
E_ents = np.load("data/E_entropies.npy")
X_imgs = generate_target_images_mini(Y_labs, E_ents)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

print("Экспресс-калибровка весов суррогата (25 эпох)...")
for epoch in range(25):
    inputs = torch.tensor(X_mats, dtype=torch.float32).to(device)
    targets = torch.tensor(X_imgs, dtype=torch.float32).to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# 4. Проверка работы модели на разрушенных состояниях
model.eval()
with torch.no_grad():
    out_orig = model(t_orig).cpu().squeeze().numpy()
    out_deph = model(t_deph).cpu().squeeze().numpy()
    out_rand = model(t_rand).cpu().squeeze().numpy()

# 5. Визуализация результатов эксперимента
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(out_orig, cmap='gray')
axes[0].set_title("1. ρ_original\n(Сохраняет когерентность)")
axes[0].axis('off')

axes[1].imshow(out_deph, cmap='gray')
axes[1].set_title("2. ρ_dephased\n(Только классическая статистика)")
axes[1].axis('off')

axes[2].imshow(out_rand, cmap='gray')
axes[2].set_title("3. ρ_randomized\n(Разрушенная статистика)")
axes[2].axis('off')

plt.tight_layout()
os.makedirs("results", exist_ok=True)
output_path = "results/controlled_destruction_test.png"
plt.savefig(output_path)
print(f"\n[УСПЕХ] Эксперимент завершен! Результат сохранен в '{output_path}'")

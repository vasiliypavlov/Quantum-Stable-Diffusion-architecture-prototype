import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)

# Архитектура 4-канального суррогата QSD
class QSD4ChannelNet(nn.Module):
    def __init__(self):
        super(QSD4ChannelNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=2, stride=2), 
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=2, stride=2), 
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.LeakyReLU(0.1)
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

    def forward(self, x):
        latent = self.encoder(x)
        projected = self.latent_projector(latent).view(-1, 64, 4, 4)
        return self.decoder(projected)

# Загрузка камуфлированных данных
X_4ch = np.load("data/X_4ch_v2.npy")
Y_labs = np.load("data/Y_labels_v2.npy")
P_phas = np.load("data/P_phases_v2.npy")

# Генерация таргетов, зависящих СТРОГО от квантовой фазы P
def generate_phase_images(Y, P):
    images = np.zeros((len(Y), 1, 64, 64))
    x, y = np.linspace(-3, 3, 64), np.linspace(-3, 3, 64)
    X_g, Y_g = np.meshgrid(x, y)
    
    for i in range(len(Y)):
        if Y[i] == 0: # Сепарабельное -> Гладкий фон (круги удалены, чтобы убрать лазейку)
            img = np.zeros((64, 64))
        else: # Запутанное -> Угол наклона зебры зависит ОТ ФАЗЫ phi!
            phi = P[i]
            img = np.sin(X_g * np.cos(phi) * 4.0 + Y_g * np.sin(phi) * 4.0)
        images[i, 0, :, :] = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return images

X_imgs = generate_phase_images(Y_labs, P_phas)

# Обучение
model = QSD4ChannelNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

inputs = torch.tensor(X_4ch, dtype=torch.float32).to(device)
targets = torch.tensor(X_imgs, dtype=torch.float32).to(device)

print("Обучение криптостойкого квантового суррогата (50 эпох)...")
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# ТЕСТ КОНТРОЛЬНОГО РАЗРУШЕНИЯ
model.eval()

# Выбираем тестовую фазу phi = 1.0 (около 57 градусов наклона)
test_phi = 1.0
psi_test = np.zeros(16, dtype=complex)
psi_test[0] = 1.0 / np.sqrt(2)
psi_test[15] = np.exp(1j * test_phi) / np.sqrt(2)
rho_orig = np.outer(psi_test, np.conj(psi_test))

# Разрушение когерентности
rho_deph = np.diag(np.diag(rho_orig))

def to_tensor_4ch(rho):
    # Извлечение компонентов внутри функции для предотвращения багов
    size = rho.shape[0]
    dm = np.eye(size, dtype=bool)
    d_re = np.zeros_like(rho, dtype=float); d_re[dm] = np.real(rho[dm])
    o_re = np.zeros_like(rho, dtype=float); o_re[~dm] = np.real(rho[~dm])
    o_im = np.zeros_like(rho, dtype=float); o_im[~dm] = np.imag(rho[~dm])
    sm = np.fromfunction(lambda i, j: np.abs(i - j) / float(size), (size, size), dtype=float)
    res = np.stack([d_re, o_re, o_im, sm], axis=0)
    return torch.tensor(res, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    out_orig = model(to_tensor_4ch(rho_orig)).cpu().squeeze().numpy()
    out_deph = model(to_tensor_4ch(rho_deph)).cpu().squeeze().numpy()

# Рендеринг финального доказательства когерентности
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(out_orig, cmap='gray')
axes[0].set_title("1. ρ_original\n(Квантовая фаза присутствует)")
axes[0].axis('off')

axes[1].imshow(out_deph, cmap='gray')
axes[1].set_title("2. ρ_dephased\n(Когерентность удалена)")
axes[1].axis('off')

plt.tight_layout()
output_path = "results/controlled_destruction_test_v3.png"
plt.savefig(output_path)
print(f"\n[УСПЕХ] Результат сохранен в '{output_path}'")

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from pauli_adapter_sd import PauliBasisDecomposer, QuantumToEmbeddingProjector

# Фиксируем сид
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Генерируем цепочку состояний с ПЛАВНО нарастающей запутанностью
# Мы смешиваем сепарабельное состояние |0000><0000| и максимально запутанное |GHZ><GHZ|
# через параметр p от 0.0 (нет запутанности) до 1.0 (максимум)
print("Синтез квантовой траектории (10 шагов нарастания запутанности)...")
decomposer = PauliBasisDecomposer()

psi_ghz = np.zeros(16, dtype=complex)
psi_ghz[0] = 1.0 / np.sqrt(2)
psi_ghz[15] = 1.0 / np.sqrt(2)
rho_max_ent = np.outer(psi_ghz, np.conj(psi_ghz)) # Максимально запутанное

psi_sep = np.zeros(16, dtype=complex)
psi_sep[0] = 1.0
rho_sep = np.outer(psi_sep, np.conj(psi_sep)) # Полностью сепарабельное

steps = 10
p_steps = np.linspace(0.0, 1.0, steps)
pauli_vectors = []

for p in p_steps:
    # Интерполяция состояний (плавное нарастание когерентности и энтропии)
    rho_current = (1.0 - p) * rho_sep + p * rho_max_ent
    c_pauli = decomposer.decompose(rho_current)
    pauli_vectors.append(c_pauli)

# Переводим в тензоры
pauli_tensor = torch.tensor(np.array(pauli_vectors), dtype=torch.float32).to(device)

# 2. Пропускаем через наш Линейный проектор [77 x 768]
projector = QuantumToEmbeddingProjector().to(device)
projector.eval()

with torch.no_grad():
    # Выходной тензор имеет форму [10, 77, 768]
    embeddings = projector(pauli_tensor)
    # Сглаживаем токены для вычисления расстояния между полными эмбеддингами: [10, 59136]
    flat_embeddings = embeddings.view(steps, -1)

# 3. Вычисляем матрицу латентных Евклидовых расстояний (10х10)
latent_distance_matrix = torch.cdist(flat_embeddings, flat_embeddings, p=2.0).cpu().numpy()

# 4. Визуализация геометрии скрытого пространства
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(latent_distance_matrix, cmap='viridis', origin='lower')
plt.colorbar(im, label='Евклидово расстояние в латентном пространстве ИИ')

ax.set_xticks(np.arange(steps))
ax.set_yticks(np.arange(steps))
ax.set_xticklabels([f"{p:.1f}" for p in p_steps])
ax.set_yticklabels([f"{p:.1f}" for p in p_steps])

ax.set_xlabel("Параметр запутанности p (Состояние 2)")
ax.set_ylabel("Параметр запутанности p (Состояние 1)")
ax.set_title("Деформация латентного пространства Stable Diffusion\nпри изменении квантовой запутанности")

# Сохранение
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "results")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "latent_geometry_test.png")
plt.tight_layout()
plt.savefig(output_path)
print(f"\n[УСПЕХ] Метрический тест завершен! График сохранен по пути:\n{output_path}")

import os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, kron, eye
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import imageio

# --- КВАНТОВЫЙ БАЗИС МОДЕЛИ ГЕЙЗЕНБЕРГА ---
X = csr_matrix([[0, 1], [1, 0]], dtype=np.complex128)
Y = csr_matrix([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = csr_matrix([[1, 0], [0, -1]], dtype=np.complex128)
I = eye(2, dtype=np.complex128)

N_QUBITS = 8
DIM = 2**N_QUBITS

def get_op(op, site, n_qubits):
    res = eye(1, dtype=np.complex128)
    for i in range(n_qubits):
        res = kron(res, op) if i == site else kron(res, I)
    return csr_matrix(res)

XX_nearest = [get_op(X, i, N_QUBITS) @ get_op(X, (i+1)%N_QUBITS, N_QUBITS) for i in range(N_QUBITS)]
YY_nearest = [get_op(Y, i, N_QUBITS) @ get_op(Y, (i+1)%N_QUBITS, N_QUBITS) for i in range(N_QUBITS)]
ZZ_nearest = [get_op(Z, i, N_QUBITS) @ get_op(Z, (i+1)%N_QUBITS, N_QUBITS) for i in range(N_QUBITS)]
Z_ops = [get_op(Z, i, N_QUBITS) for i in range(N_QUBITS)]

def compute_linear_entropy(psi):
    psi_reshaped = psi.reshape(2, 128)
    rho_A = psi_reshaped @ psi_reshaped.conj().T
    return 1.0 - np.trace(rho_A @ rho_A).real

def get_heisenberg_anisotropic_data(J_xy, J_z):
    H = csr_matrix((DIM, DIM), dtype=np.complex128)
    for i in range(N_QUBITS):
        H -= J_xy * (XX_nearest[i] + YY_nearest[i])
        H -= J_z * ZZ_nearest[i]
    _, eigenvectors = spla.eigsh(H, k=1, which='SA')
    psi = eigenvectors[:, 0]
    
    features = []
    for i in range(N_QUBITS): features.append(np.vdot(psi, Z_ops[i].dot(psi)).real)
    for i in range(N_QUBITS): features.append(np.vdot(psi, XX_nearest[i].dot(psi)).real)
    for i in range(N_QUBITS): features.append(np.vdot(psi, YY_nearest[i].dot(psi)).real)
    for i in range(N_QUBITS): features.append(np.vdot(psi, ZZ_nearest[i].dot(psi)).real)
    return np.array(features, dtype=np.float32), compute_linear_entropy(psi)

# --- АРХИТЕКТУРА ПРОЕКТОРА ДЛЯ ОБУЧЕНИЯ ---
class HeisenbergQuantumProjector(nn.Module):
    def __init__(self, input_dim=32, seq_len=77, embed_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, seq_len * embed_dim)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, self.seq_len, self.embed_dim)
        return self.layer_norm(out)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("animation_real_projector", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Загрузка Stable Diffusion
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None
    try: pipe.enable_xformers_memory_efficient_attention()
    except Exception: pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()

    # 2. Подготовка физической траектории
    steps = 60
    J_xy_space = np.linspace(0.1, 2.5, steps)
    J_z_space = np.linspace(2.5, 0.1, steps)

    print("Генерация физических данных...")
    corr_history, entropy_history = [], []
    for J_xy, J_z in zip(J_xy_space, J_z_space):
        corr, ent = get_heisenberg_anisotropic_data(J_xy, J_z)
        corr_history.append(corr)
        entropy_history.append(ent)

    # 3. ПОЛУЧЕНИЕ НАСТОЯЩИХ ЭМБЕДДИНГОВ CLIP ДЛЯ ОБУЧЕНИЯ
    print("\nЭкстракция текстовых эмбеддингов из CLIP...")
    prompt_order = "perfect crystal lattice, hyper-ordered structure, geometric symmetry, solid cold monolith"
    prompt_chaos = "quantum chaos, high entropy foam, turbulent fluid plasma, cosmic soup"

    def get_clip_embeds(prompt):
        tokens = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        with torch.no_grad():
            return pipe.text_encoder(tokens.input_ids.to(device)).last_hidden_state.squeeze(0)

    embed_order = get_clip_embeds(prompt_order)
    embed_chaos = get_clip_embeds(prompt_chaos)

    # 4. ОБУЧЕНИЕ ПРОЕКТОРА «ПО-НАСТОЯЩЕМУ»
    print("\nИнициализация и запуск обучения проектора...")
    projector = HeisenbergQuantumProjector().to(device)
    optimizer = optim.AdamW(projector.parameters(), lr=5e-4, weight_decay=1e-2)
    criterion = nn.MSELoss()

    # Формируем обучающий датасет на основе вычисленной энтропии
    X_train = torch.tensor(np.array(corr_history), dtype=torch.float32).to(device)
    Y_train_list = []
    
    for ent in entropy_history:
        # Нормализуем вес энтропии (примерно от 0 до 0.5) в интервал [0, 1]
        weight_chaos = np.clip(ent / 0.5, 0.0, 1.0)
        weight_order = 1.0 - weight_chaos
        # Интерполируем целевой эмбеддинг CLIP
        target_embed = weight_order * embed_order + weight_chaos * embed_chaos
        Y_train_list.append(target_embed)
        
    Y_train = torch.stack(Y_train_list).to(device)

    # Цикл оптимизации весов (150 эпох для точной подгонки на лету)
    projector.train()
    for epoch in range(1, 151):
        optimizer.zero_grad()
        outputs = projector(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            print(f"Эпоха {epoch}/150 | Ошибка проекции (MSE Loss): {loss.item():.6f}")

    # Переводим проектор в режим инференса и FP16
    projector.eval()
    projector.to(torch.float16)

    # 5. ИНФЕРЕНС С ОБУЧЕННЫМ ПРОЕКТОРОМ
    print("\nЗапуск генерации кадров с обученным квантовым инжектором...")
    frame_paths = []
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Calibri', 'DejaVu Sans']

    for idx in range(steps):
        J_xy, J_z = J_xy_space[idx], J_z_space[idx]
        current_entropy = entropy_history[idx]
        print(f"Кадр {idx+1}/{steps} | Обученный Проектор | Энтропия S_L={current_entropy:.4f}")
        
        quantum_tensor = torch.tensor(corr_history[idx], dtype=torch.float16).unsqueeze(0).to(device)
        with torch.no_grad():
            # Теперь проектор сам знает, как выдать правильный семантический эмбеддинг!
            trained_embeddings = projector(quantum_tensor)

        generator = torch.Generator(device=device).manual_seed(42)
        torch.cuda.empty_cache()
        
        output = pipe(
            prompt_embeds=trained_embeddings, 
            num_inference_steps=20,
            guidance_scale=8.0,
            width=512,
            height=512,
            generator=generator
        )
         # Явно вытаскиваем первую картинку из объекта вывода
        sd_image = output.images[0]
        if not isinstance(sd_image, Image.Image):
            # Если это всё ещё не чистый PIL Image (например, тензор или массив)
            sd_image = Image.fromarray(np.uint8(sd_image))

        # Отрисовка графика
        fig, ax = plt.subplots(figsize=(5, 5.12), dpi=100)
        ax.plot(range(steps), entropy_history, color='lime', alpha=0.5, label='Энтропия S_L (Trained)')
        ax.scatter(idx, current_entropy, color='red', s=100, zorder=5, label='Текущая фаза')
        
        ax.set_title(f"Обученный квантовый инжектор\nШаг {idx+1}: J_xy={J_xy:.2f}, J_z={J_z:.2f}", fontsize=10, color='white')
        ax.set_xticks(np.linspace(0, steps-1, 5))
        ax.set_xticklabels([f"{j:.1f}" for j in np.linspace(0.1, 2.5, 5)])
        ax.set_xlabel("Параметр взаимодействия J_xy", fontsize=8)
        ax.set_ylabel("Линейная Энтропия (S_L)", fontsize=8)
        ax.set_ylim(-0.05, 0.55)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='lower left', fontsize=8)
        
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        plot_image = Image.frombuffer('RGBA', fig.canvas.get_width_height(), rgba_buffer, 'raw', 'RGBA', 0, 1).convert('RGB')
        plt.close(fig)

        # Склейка
        combined_image = Image.new('RGB', (1012, 512))
        combined_image.paste(sd_image, (0, 0))
        combined_image.paste(plot_image, (512, 0))

        frame_path = f"animation_real_projector/frame_{idx:03d}.png"
        combined_image.save(frame_path)
        frame_paths.append(frame_path)

    print("\nСборка финального видеоролика...")
    video_path = "results/heisenberg_8q_real_trained.mp4"
    images = [imageio.v2.imread(f) for f in frame_paths]
    imageio.mimsave(video_path, images, fps=12, macro_block_size=16)
    print(f"[УСПЕХ] Эксперимент завершен. Видео сохранено в: {video_path}")

if __name__ == "__main__":
    main()

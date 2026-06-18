import os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, kron, eye
import torch.nn as nn
import imageio
import matplotlib.pyplot as plt
from PIL import Image

# --- ИСПРАВЛЕННЫЕ МАТРИЦЫ ПАУЛИ (Учтен опыт прошлых итераций!) ---
X = csr_matrix([[0, 1], [1, 0]], dtype=np.complex128)
Z = csr_matrix([[1, 0], [0, -1]], dtype=np.complex128)
I = eye(2, dtype=np.complex128)

# --- КВАНТОВЫЙ ПРОЕКТОР НА ФИЗИЧЕСКИХ КОРРЕЛЯЦИЯХ ---
class PhysicalQuantumProjector(nn.Module):
    def __init__(self, input_dim=36, seq_len=77, embed_dim=768):
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

# --- ВЫЧИСЛИТЕЛЬНЫЙ КВАНТОВЫЙ ДВИЖОК С ЛИНЕЙНОЙ ЭНТРОПИЕЙ ---
N_QUBITS = 8
DIM = 2**N_QUBITS

def get_op(op, site, n_qubits):
    res = eye(1, dtype=np.complex128)
    for i in range(n_qubits):
        res = kron(res, op) if i == site else kron(res, I)
    return csr_matrix(res)

X_ops = [get_op(X, i, N_QUBITS) for i in range(N_QUBITS)]
ZZ_ops = []
for i in range(N_QUBITS):
    for j in range(i + 1, N_QUBITS):
        ZZ_ops.append(get_op(Z, i, N_QUBITS) @ get_op(Z, j, N_QUBITS))

def compute_linear_entropy(psi):
    """
    Вычисляет Линейную энтропию S_L для первого кубита подсистемы
    методом частичного следа (Partial Trace) матрицы плотности rho
    """
    # Решейпим вектор состояния под разделение: кубит_0 (размер 2) и остальной регистр (размер 128)
    psi_reshaped = psi.reshape(2, 128)
    # rho_A = Partial Trace по остальным кубитам
    rho_A = psi_reshaped @ psi_reshaped.conj().T
    # S_L = 1 - Tr(rho_A^2)
    rho_A_sq = rho_A @ rho_A
    tr_rho_sq = np.trace(rho_A_sq).real
    return 1.0 - tr_rho_sq

def get_quantum_data(J, g):
    """Возвращает физические корреляции и точное значение линейной энтропии"""
    H = csr_matrix((DIM, DIM), dtype=np.complex128)
    for i in range(N_QUBITS):
        H -= J * (get_op(Z, i, N_QUBITS) @ get_op(Z, (i + 1) % N_QUBITS, N_QUBITS))
    for i in range(N_QUBITS):
        H -= g * X_ops[i]
        
    _, eigenvectors = spla.eigsh(H, k=1, which='SA')
    psi = eigenvectors[:, 0]
    
    # 1. Корреляции спинов для проектора
    features = []
    for op in X_ops:
        features.append(np.vdot(psi, op.dot(psi)).real)
    for op in ZZ_ops:
        features.append(np.vdot(psi, op.dot(psi)).real)
        
    # 2. Вычисление энтропии запутанности
    entropy = compute_linear_entropy(psi)
    
    return np.array(features, dtype=np.float32), entropy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("animation_frames_plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Инициализация Stable Diffusion с оптимизациями под RTX 3070
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()

    # 2. Инициализация проектора
    projector = PhysicalQuantumProjector().to(device).to(torch.float16)
    projector.eval()

    # 3. Текстовый стиль
    style_prompt = "quantum physics visualization, abstract complex fractal, scientific pattern, high entropy math texture"
    text_inputs = pipe.tokenizer(style_prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_outputs = pipe.text_encoder(text_inputs.input_ids.to(device))
        text_embeddings = text_outputs.last_hidden_state

    # 4. Траектория параметров
    steps = 60  
    J_space = np.linspace(3.0, 0.0, steps)
    g_space = np.linspace(0.0, 3.0, steps)

    # Предварительно посчитаем значения энтропии по всей траектории для построения графика
    print("Предварительный расчет физических траекторий...")
    corr_history = []
    entropy_history = []
    for J, g in zip(J_space, g_space):
        corr, ent = get_quantum_data(J, g)
        corr_history.append(corr)
        entropy_history.append(ent)

    print(f"\nЗапуск генерации {steps} кадров со встроенной аналитикой...")
    frame_paths = []

    plt.style.use('dark_background') # Научный темный стиль под стать фракталам

    for idx in range(steps):
        J, g = J_space[idx], g_space[idx]
        current_entropy = entropy_history[idx]
        print(f"Кадр {idx+1}/{steps} | Вычисление кадра для J={J:.2f}, g={g:.2f} | Энтропия S_L={current_entropy:.4f}")
        
        # Инжекция в нейросеть
        quantum_tensor = torch.tensor(corr_history[idx], dtype=torch.float16).unsqueeze(0).to(device)
        with torch.no_grad():
            quantum_embeddings = projector(quantum_tensor)
            mixed_embeddings = 0.55 * quantum_embeddings + 0.45 * text_embeddings

        generator = torch.Generator(device=device).manual_seed(42)
        torch.cuda.empty_cache()
        
        output = pipe(
            prompt_embeds=mixed_embeddings,
            num_inference_steps=20,
            guidance_scale=8.0,
            width=512,
            height=512,
            generator=generator
        )
        
        # Исправлено получение одиночного изображения через [0]
        sd_image = output.images[0] 

        # --- РЕНДЕРИНГ ГРАФИКА ДИНАМИКИ ЧЕРЕЗ MATPLOTLIB ---
        fig, ax = plt.subplots(figsize=(5, 5.12), dpi=100)
        ax.plot(range(steps), entropy_history, color='cyan', alpha=0.4, label='Траектория S_L')
        # Ставим яркую пульсирующую красную точку на текущий шаг
        ax.scatter(idx, current_entropy, color='red', s=100, zorder=5, label='Текущее состояние')
        
        ax.set_title(f"Квантовые параметры (8 кубит)\nШаг {idx+1}: J={J:.2f}, g={g:.2f}", fontsize=10, color='white')
        ax.set_xlabel("Шаг эволюции системы", fontsize=8)
        ax.set_ylabel("Линейная Энтропия (S_L)", fontsize=8)
        ax.set_ylim(-0.05, 0.55)
        ax.set_xticks(np.linspace(0, steps-1, 5))
        ax.set_xticklabels([f"{g:.1f}" for g in np.linspace(0.0, 3.0, 5)])
        ax.set_xlabel("Поперечное поле (g)", fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        
        # --- ИСПРАВЛЕННЫЙ И ЧЕТКИЙ РЕНДЕРИНГ ГРАФИКА ---
        fig.canvas.draw()
        
        # Получаем чистый RGBA буфер памяти без искажения сдвига байт
        rgba_buffer = fig.canvas.buffer_rgba()
        
        # Конвертируем строго в формат 'RGBA', чтобы Pillow прочитал альфа-канал
        plot_image = Image.frombuffer('RGBA', fig.canvas.get_width_height(), rgba_buffer, 'raw', 'RGBA', 0, 1)
        
        # Переводим обратно в RGB для склейки с фракталом
        plot_image = plot_image.convert('RGB')
        plt.close(fig)

        # --- СКЛЕЙКА ФРАКТАЛА И ГРАФИКА (БОК О БОК) ---
        combined_image = Image.new('RGB', (1012, 512))
        combined_image.paste(sd_image, (0, 0))
        combined_image.paste(plot_image, (512, 0))

        frame_path = f"animation_frames_plots/frame_{idx:03d}.png"
        combined_image.save(frame_path)
        frame_paths.append(frame_path)

    # 5. Сборка видеоролика
    print("\nСборка финального аналитического видео...")
    video_path = "results/ising_8q_with_metrics.mp4"
    images = [imageio.v2.imread(f) for f in frame_paths]
    imageio.mimsave(video_path, images, fps=12, macro_block_size=16)
    print(f"\n[УСПЕХ] Готово! Научная анимация с графиком энтропии сохранена: {video_path}")

if __name__ == "__main__":
    main()

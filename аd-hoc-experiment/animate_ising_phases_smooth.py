import os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, kron, eye
import torch.nn as nn
import imageio

# --- ОБНОВЛЕННЫЙ ПРОЕКТОР ПОД ВЕРОЯТНОСТНЫЙ ВХОД (256 вместо 512) ---
class SmoothQuantumProjector(nn.Module):
    def __init__(self, input_dim=256, seq_len=77, embed_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # Мощная, но легкая сверточная/линейная сеть для сглаживания распределений
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, seq_len * embed_dim)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, self.seq_len, self.embed_dim)
        return self.layer_norm(out)

# --- КВАНТОВЫЙ ДВИЖОК (8 КУБИТ) ---
N_QUBITS = 8
DIM = 2**N_QUBITS
X = csr_matrix([[0, 1], [1, 0]])
Z = csr_matrix([[1, 0], [0, -1]])
I = eye(2)

def get_op(op, site, n_qubits):
    res = eye(1)
    for i in range(n_qubits):
        res = kron(res, op) if i == site else kron(res, I)
    return csr_matrix(res)

def get_smooth_quantum_features(J, g):
    """
    Генерирует профиль вероятностей состояния, полностью очищенный
    от хаотичных прыжков знаков комплексной фазы.
    """
    H = csr_matrix((DIM, DIM))
    for i in range(N_QUBITS):
        H -= J * (get_op(Z, i, N_QUBITS) @ get_op(Z, (i + 1) % N_QUBITS, N_QUBITS))
    for i in range(N_QUBITS):
        H -= g * get_op(X, i, N_QUBITS)
        
    _, eigenvectors = spla.eigsh(H, k=1, which='SA')
    psi = eigenvectors[:, 0]
    
    # ХАК СТАБИЛИЗАЦИИ: Берем квадрат модуля. Распределение плотности 
    # квантовых состояний строго непрерывно при фазовом переходе!
    probabilities = np.abs(psi) ** 2
    return probabilities.astype(np.float32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("animation_frames_smooth", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Загрузка и оптимизация Stable Diffusion
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()

    # 2. Инициализация адаптированного проектора
    projector = SmoothQuantumProjector().to(device).to(torch.float16)
    projector.eval()

    # 3. Текстовый стабилизатор стиля
    style_prompt = "quantum physics visualization, abstract complex fractal, scientific pattern, high entropy math texture"
    text_inputs = pipe.tokenizer(style_prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_outputs = pipe.text_encoder(text_inputs.input_ids.to(device))
        text_embeddings = text_outputs.last_hidden_state

    # 4. Параметры траектории эволюции
    steps = 60  
    J_space = np.linspace(2.5, 0.0, steps)  # Увеличили верхний порог J для более явного порядка
    g_space = np.linspace(0.0, 2.5, steps)

    print(f"\nЗапуск ГЛАДКОЙ генерации {steps} состояний...")
    frame_paths = []

    for idx, (J, g) in enumerate(zip(J_space, g_space)):
        print(f"Кадр {idx+1}/{steps} | Расчет физики для J={J:.2f}, g={g:.2f}")
        
        # Получаем чистый инвариантный вектор вероятностей
        prob_vector = get_smooth_quantum_features(J, g)
        quantum_tensor = torch.tensor(prob_vector, dtype=torch.float16).unsqueeze(0).to(device)
        
        with torch.no_grad():
            quantum_embeddings = projector(quantum_tensor)
            
            # Смешиваем эмбеддинги. Начнем с веса 0.25 для квантовой составляющей, 
            # чтобы текстовый стиль жестче удерживал непрерывность геометрии
            mixed_embeddings = 0.25 * quantum_embeddings + 0.75 * text_embeddings

        # КРИТИЧЕСКИЙ ФАКТОР: Фиксируем один и тот же сид для всего процесса
        generator = torch.Generator(device=device).manual_seed(101)
        
        torch.cuda.empty_cache()
        output = pipe(
            prompt_embeds=mixed_embeddings,
            num_inference_steps=20,
            guidance_scale=10.0, # Увеличили масштаб, чтобы жестче следовать плавному эмбеддингу
            width=512,
            height=512,
            generator=generator
        )
        
        image = output.images[0]
        frame_path = f"animation_frames_smooth/frame_{idx:03d}.png"
        image.save(frame_path)
        frame_paths.append(frame_path)

    # 5. Сборка видео
    print("\nСклеивание плавной квантовой анимации...")
    video_path = "results/ising_8q_smooth_evolution.mp4"
    images = [imageio.v2.imread(f) for f in frame_paths]
    imageio.mimsave(video_path, images, fps=15, macro_block_size=16)
    print(f"[УСПЕХ] Плавная эволюция фаз зафиксирована: {video_path}")

if __name__ == "__main__":
    main()

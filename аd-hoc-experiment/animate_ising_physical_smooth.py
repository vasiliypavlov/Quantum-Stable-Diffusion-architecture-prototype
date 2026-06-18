import os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, kron, eye
import torch.nn as nn
import imageio

# --- КВАНТОВЫЙ ПРОЕКТОР НА ФИЗИЧЕСКИХ КОРРЕЛЯЦИЯХ (Вход: 36 признаков) ---
class PhysicalQuantumProjector(nn.Module):
    def __init__(self, input_dim=36, seq_len=77, embed_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # Глубокая сеть для развертывания компактных физических корреляций в CLIP-пространство
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

# --- КВАНТОВЫЙ ДВИЖОК ОЖИДАЕМЫХ ЗНАЧЕНИЙ (8 КУБИТ) ---
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

# Предрасчет операторов для ускорения цикла
X_ops = [get_op(X, i, N_QUBITS) for i in range(N_QUBITS)]
ZZ_ops = []
for i in range(N_QUBITS):
    for j in range(i + 1, N_QUBITS):
        ZZ_ops.append(get_op(Z, i, N_QUBITS) @ get_op(Z, j, N_QUBITS))

def get_physical_correlations(J, g):
    """
    Рассчитывает физические наблюдаемые (корреляции Спинов).
    Они строго непрерывны, инвариантны к знаку фазы и плавно меняются 
    от ферромагнитного порядка к парамагнитному хаосу.
    """
    H = csr_matrix((DIM, DIM))
    # Ближайшие соседи для гамильтониана Изинга
    for i in range(N_QUBITS):
        H -= J * (get_op(Z, i, N_QUBITS) @ get_op(Z, (i + 1) % N_QUBITS, N_QUBITS))
    for i in range(N_QUBITS):
        H -= g * X_ops[i]
        
    _, eigenvectors = spla.eigsh(H, k=1, which='SA')
    psi = eigenvectors[:, 0]
    
    features = []
    # 1. Намагниченность по полю <X_i> (8 признаков)
    for op in X_ops:
        val = np.vdot(psi, op.dot(psi))
        features.append(val.real)
        
    # 2. Пространственные корреляции <Z_i Z_j> (28 признаков)
    for op in ZZ_ops:
        val = np.vdot(psi, op.dot(psi))
        features.append(val.real)
        
    return np.array(features, dtype=np.float32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("animation_frames_physical", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Инициализация Stable Diffusion
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()

    # 2. Инициализация проектора корреляций
    projector = PhysicalQuantumProjector().to(device).to(torch.float16)
    projector.eval()

    # 3. Базовый текстовый стиль
    style_prompt = "quantum physics visualization, abstract complex fractal, scientific pattern, high entropy math texture"
    text_inputs = pipe.tokenizer(style_prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_outputs = pipe.text_encoder(text_inputs.input_ids.to(device))
        text_embeddings = text_outputs.last_hidden_state

    # 4. Диапазон эволюции параметров
    steps = 60  
    J_space = np.linspace(3.0, 0.0, steps) # Расширили диапазон для глубокого фазового контраста
    g_space = np.linspace(0.0, 3.0, steps)

    print(f"\nЗапуск физически плавной генерации ({steps} кадров)...")
    frame_paths = []

    for idx, (J, g) in enumerate(zip(J_space, g_space)):
        print(f"Кадр {idx+1}/{steps} | Вычисление спиновых корреляций для J={J:.2f}, g={g:.2f}")
        
        # Вычисляем вектор наблюдаемых
        corr_vector = get_physical_correlations(J, g)
        quantum_tensor = torch.tensor(corr_vector, dtype=torch.float16).unsqueeze(0).to(device)
        
        with torch.no_grad():
            quantum_embeddings = projector(quantum_tensor)
            # СДВИГАЕМ БАЛАНС: 55% квантовой физики, 45% текста стиля для масштабных изменений геометрии
            mixed_embeddings = 0.55 * quantum_embeddings + 0.45 * text_embeddings

        generator = torch.Generator(device=device).manual_seed(42)
        
        torch.cuda.empty_cache()
        output = pipe(
            prompt_embeds=mixed_embeddings,
            num_inference_steps=20,
            guidance_scale=8.0, # Ослабили привязку, позволяя геометрии течь свободнее
            width=512,
            height=512,
            generator=generator
        )
        
        image = output.images[0]
        frame_path = f"animation_frames_physical/frame_{idx:03d}.png"
        image.save(frame_path)
        frame_paths.append(frame_path)

    # 5. Сборка видео
    print("\nСборка финального видео...")
    video_path = "results/ising_8q_physical_evolution.mp4"
    images = [imageio.v2.imread(f) for f in frame_paths]
    imageio.mimsave(video_path, images, fps=12, macro_block_size=16)
    print(f"\n[УСПЕХ] Готово! Истинная эволюция квантовых корреляций сохранена в: {video_path}")

if __name__ == "__main__":
    main()

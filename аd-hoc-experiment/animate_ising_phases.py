import os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, kron, eye
from train_adhoc_projector import AdHocMultiChannelProjector
import imageio

# --- КВАНТОВЫЙ БАЗИС (8 КУБИТ) ---
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

def get_ground_state(J, g):
    H = csr_matrix((DIM, DIM))
    for i in range(N_QUBITS):
        H -= J * (get_op(Z, i, N_QUBITS) @ get_op(Z, (i + 1) % N_QUBITS, N_QUBITS))
    for i in range(N_QUBITS):
        H -= g * get_op(X, i, N_QUBITS)
    _, eigenvectors = spla.eigsh(H, k=1, which='SA')
    psi = eigenvectors[:, 0]
    return np.stack([psi.real, psi.imag], axis=-1).flatten()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("animation_frames", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Загрузка и оптимизация SD для RTX 3070
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()

    # 2. Инициализация проектора
    projector = AdHocMultiChannelProjector().to(device).to(torch.float16)
    projector.eval()

    # 3. Подготовка текстового якоря стиля
    style_prompt = "quantum physics visualization, abstract complex fractal, scientific pattern, high entropy math texture"
    text_inputs = pipe.tokenizer(style_prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        text_outputs = pipe.text_encoder(text_inputs.input_ids.to(device))
        text_embeddings = text_outputs.last_hidden_state

    # 4. Траектория изменения параметров — УВЕЛИЧИЛИ ДО 60 ШАГОВ
    steps = 60  
    J_space = np.linspace(2.0, 0.1, steps)
    g_space = np.linspace(0.0, 2.0, steps)

    print(f"\nЗапуск генерации {steps} фазовых состояний для ультра-плавной анимации...")
    
    frame_paths = []

    for idx, (J, g) in enumerate(zip(J_space, g_space)):
        print(f"Генерация кадра {idx+1}/{steps} (J={J:.2f}, g={g:.2f})...")
        
        # Расчет квантового состояния
        psi_vector = get_ground_state(J, g)
        quantum_tensor = torch.tensor(psi_vector, dtype=torch.float16).unsqueeze(0).to(device)
        
        with torch.no_grad():
            quantum_embeddings = projector(quantum_tensor)
            mixed_embeddings = 0.3 * quantum_embeddings + 0.7 * text_embeddings

        # Фиксируем генератор шума для плавности анимации
        generator = torch.Generator(device=device).manual_seed(42)
        
        torch.cuda.empty_cache()
        output = pipe(
            prompt_embeds=mixed_embeddings,
            num_inference_steps=20, # Чуть снизили шаги диффузии (с 25 до 20) для ускорения без потери качества
            guidance_scale=9.0,
            width=512,
            height=512,
            generator=generator
        )
        
        image = output.images[0]
        frame_path = f"animation_frames/frame_{idx:03d}.png"
        image.save(frame_path)
        frame_paths.append(frame_path)

    # 5. АВТОМАТИЧЕСКАЯ СБОРКА ВИДЕО И GIF
    print("\nСборка плавного видеоролика...")
    video_path = "results/ising_8q_evolution.mp4"
    gif_path = "results/ising_8q_evolution.gif"
    
    # Читаем кадры
    images = [imageio.imread(f) for f in frame_paths]
    
    # Сохраняем MP4 видео со скоростью 15 кадров в секунду (4 секунды идеальной эволюции)
    imageio.mimsave(video_path, images, fps=15, macro_block_size=16)
    # Дублируем в GIF для удобного шаринга
    imageio.mimsave(gif_path, images, fps=15)
    
    print(f"Успех! Видео сохранено в: {video_path}")
    print(f"Анимация GIF сохранена в: {gif_path}")

if __name__ == "__main__":
    main()

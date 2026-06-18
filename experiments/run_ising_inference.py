import torch
import torch.nn as nn
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw
import os
import gc

class MultiChannelQuantumProjector(nn.Module):
    def __init__(self, phase_dim=64, chaos_dim=192, tokens=77, embed_dim=768):
        super().__init__()
        self.tokens = tokens
        self.embed_dim = embed_dim
        self.phase_encoder = nn.Sequential(nn.Linear(phase_dim, 256), nn.LayerNorm(256), nn.GELU())
        self.chaos_encoder = nn.Sequential(nn.Linear(chaos_dim, 256), nn.LayerNorm(256), nn.GELU())
        self.bridge = nn.Sequential(
            nn.Linear(256 + 256, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(1024, tokens * embed_dim)
        )
    def forward(self, phase_vec, chaos_vec):
        f_p = self.phase_encoder(phase_vec)
        f_c = self.chaos_encoder(chaos_vec)
        return self.bridge(torch.cat((f_p, f_c), dim=-1)).view(-1, self.tokens, self.embed_dim)

def run_ising_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Запуск промышленного Изинговского инференса на: {device}")
    
    os.makedirs("results", exist_ok=True)
    
    # 1. Загрузка классической SD 1.5
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.enable_attention_slicing()

    # 2. Загрузка обученного промышленного инжектора из папки data/
    projector = MultiChannelQuantumProjector().to(device)
    weights_path = os.path.join("data", "ising_quantum_projector.pt")
    projector.load_state_dict(torch.load(weights_path, map_location=device))
    projector.eval()
    
    # 3. Загружаем бинарный датасет для поиска реальных физических состояний
    data_path = os.path.join("data", "quantum_ising_50k.npy")
    pauli_vectors = np.load(data_path)
    
    # Считаем энтропию подсистемы для выбора эталонных кадров
    pauli_indices_A = []
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    if k == 0 and l == 0: pauli_indices_A.append(idx)
                    idx += 1
                    
    print("Поиск контрольных физических состояний в датасете...", flush=True)
    entropies = []
    for vec in pauli_vectors[:5000]: # Ищем по первой выборке для скорости
        tr_rho_A_sq = np.sum(vec[pauli_indices_A] ** 2) / 4.0
        entropies.append(1.0 - tr_rho_A_sq)
    entropies = np.array(entropies)
    
    # Выбираем индексы трех фундаментальных состояний
    idx_order = np.argmin(entropies) # Минимальная энтропия (Кристаллический порядок)
    idx_chaos = np.argmax(entropies) # Максимальная запутанность (Квантовая пена хаоса)
    
    # Точка квантового фазового перехода (критическая область в районе медианы ~0.32)
    idx_transition = np.argmin(np.abs(entropies - 0.321)) 
    
    selected_states = [
        (pauli_vectors[idx_order], f"Ising Order (S_L: {entropies[idx_order]:.3f})"),
        (pauli_vectors[idx_transition], f"Ising Phase Transition (S_L: {entropies[idx_transition]:.3f})"),
        (pauli_vectors[idx_chaos], f"Ising Quantum Chaos (S_L: {entropies[idx_chaos]:.3f})")
    ]
    
    generated_images = []
    # Фиксируем сид для жесткого контроля геометрии
    generator = torch.Generator(device=device).manual_seed(42)
    
    print("\n--- Визуализация физических фаз модели Изинга ---")
    for vec, label in selected_states:
        print(f"Рендеринг состояния: {label}...")
        
        # Нарезаем вектор на два физических канала
        p_vec = torch.tensor(vec[:64]).unsqueeze(0).to(device)
        c_vec = torch.tensor(vec[64:]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = projector(p_vec, c_vec).half()
            output = pipe(prompt_embeds=emb, num_inference_steps=25, guidance_scale=7.5, generator=generator)
            generated_images.append((output.images[0], label))
            
        gc.collect()
        torch.cuda.empty_cache()

    # Сборка триптиха Изинга в папку results/
    width, height = generated_images[0][0].size
    triptych = Image.new('RGB', (width * 3, height + 50), color=(10, 10, 10))
    
    draw = ImageDraw.Draw(triptych)
    for idx, (img, label) in enumerate(generated_images):
        triptych.paste(img, (idx * width, 0))
        draw.text((idx * width + 20, height + 15), label, fill=(240, 240, 240))
        
    output_path = os.path.join("results", "ising_phases_triptych.png")
    triptych.save(output_path)
    print(f"\n[УСПЕХ]: Физический триптих Изинга сохранен в '{output_path}'!")

if __name__ == "__main__":
    run_ising_inference()

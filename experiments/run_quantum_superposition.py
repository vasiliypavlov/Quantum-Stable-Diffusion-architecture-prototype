import torch
import torch.nn as nn
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw
import os
import gc

# 1. Многоканальная архитектура
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

def run_superposition_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Создаем целевые директории
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 2. Загрузка компонентов
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.enable_attention_slicing()

    projector = MultiChannelQuantumProjector().to(device)
    weights_path = os.path.join("data", "multichannel_quantum_projector.pt")
    
    # Автоперенос весов в нужную папку, если они лежали в корне
    if not os.path.exists(weights_path) and os.path.exists("multichannel_quantum_projector.pt"):
        os.rename("multichannel_quantum_projector.pt", weights_path)
        
    projector.load_state_dict(torch.load(weights_path, map_location=device))
    projector.eval()
    
    # 3. Создаем два базовых чистых квантовых состояния
    np.random.seed(200)
    
    # Состояние А (Низкая энтропия — сильный упорядоченный базис, фаза 0°)
    phase_A = np.zeros(64, dtype=np.float32)
    phase_A[0] = np.cos(0.0).astype(np.float32)  # ФИКС: Пишем в индекс
    phase_A[1] = np.sin(0.0).astype(np.float32)  # ФИКС: Пишем в индекс
    chaos_A = np.random.normal(0.1, 0.02, 192).astype(np.float32)
    
    # Состояние Б (Высокая энтропия — хаотический базис, фаза 90°)
    phase_B = np.zeros(64, dtype=np.float32)
    phase_B[0] = np.cos(np.pi/2).astype(np.float32)  # ФИКС: Пишем в индекс
    phase_B[1] = np.sin(np.pi/2).astype(np.float32)  # ФИКС: Пишем в индекс
    chaos_B = np.random.normal(0.8, 0.1, 192).astype(np.float32)
    
    # Смешиваем в пропорции 50/50 на уровне квантовых амплитуд (Суперпозиция \psi = c_A|A> + c_B|B>)
    c_A, c_B = 0.707, 0.707 
    
    phase_superposition = c_A * phase_A + c_B * phase_B
    chaos_superposition = c_A * chaos_A + c_B * chaos_B
    
    states_to_render = [
        (phase_A, chaos_A, "State A: Pure Order (0 deg)"),
        (phase_superposition, chaos_superposition, "State A + B: Quantum Superposition"),
        (phase_B, chaos_B, "State B: Pure Chaos (90 deg)")
    ]
    
    generated_images = []
    
    print("\n--- Запуск квантовой интерференции концептов ---")
    for idx, (p_vec, c_vec, label) in enumerate(states_to_render):
        print(f"Рендеринг: {label}...")
        gen = torch.Generator(device=device).manual_seed(42)
        
        t_p = torch.tensor(p_vec).unsqueeze(0).to(device)
        t_c = torch.tensor(c_vec).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embeds = projector(t_p, t_c).half()
            output = pipe(prompt_embeds=embeds, num_inference_steps=25, guidance_scale=7.5, generator=gen)
            img_result = output.images[0]  # ФИКС: извлекаем PIL-картинку напрямую
            generated_images.append((img_result, label))
            
        gc.collect()
        torch.cuda.empty_cache()

    # Сборка триптиха
    width, height = generated_images[0][0].size  # ФИКС: корректный замер размера первого PIL-объекта
    triptych = Image.new('RGB', (width * 3, height + 50), color=(15, 15, 15))
    
    draw = ImageDraw.Draw(triptych)
    for idx, (img, label) in enumerate(generated_images):
        triptych.paste(img, (idx * width, 0))
        draw.text((idx * width + 20, height + 15), label, fill=(230, 230, 230))
        
    output_path = os.path.join("results", "quantum_superposition_triptych.png")
    triptych.save(output_path)
    print(f"\n[Успех]: Результат интерференции сохранен в '{output_path}'!")

if __name__ == "__main__":
    run_superposition_experiment()

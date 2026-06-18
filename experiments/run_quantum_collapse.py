import torch
import torch.nn as nn
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw
import gc

class QuantumToEmbeddingProjector(nn.Module):
    def __init__(self, input_dim=256, tokens=77, embed_dim=768):
        super().__init__()
        self.tokens = tokens
        self.embed_dim = embed_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(512, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(1024, tokens * embed_dim)
        )
    def forward(self, x):
        return self.network(x).view(x.size(0), self.tokens, self.embed_dim)

def simulate_real_collapse():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Запуск исправленной симуляции коллапса на: {device}")
    
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.enable_attention_slicing()

    projector = QuantumToEmbeddingProjector().to(device)
    projector.load_state_dict(torch.load("data/physical_quantum_projector.pt", map_location=device))
    projector.eval()
    
    # Инициализируем идентичную высокоэнтропийную основу (S_L -> 1)
    np.random.seed(101)
    base_vector = np.zeros(256, dtype=np.float32)
    base_vector[:64] = np.random.normal(0.1, 0.05, 64)
    base_vector[64:] = np.random.normal(0.8, 0.1, 192)
    
    # --- СОСТОЯНИЕ №1: Чистая суперпозиция (Phase: 45°) ---
    print("\n[Шаг 1/2] Генерация когерентного состояния (45°)...")
    phase_45 = np.pi / 4
    vector_clean = base_vector.copy()
    
    # ИСПРАВЛЕНИЕ: Записываем фазу строго в выделенные ячейки памяти
    vector_clean[42] = np.cos(phase_45).astype(np.float32)
    vector_clean[43] = np.sin(phase_45).astype(np.float32)
    
    generator = torch.Generator(device=device).manual_seed(42)
    input_clean = torch.tensor(vector_clean).unsqueeze(0).to(device)
    with torch.no_grad():
        embeds_clean = projector(input_clean).half()
        img_clean = pipe(prompt_embeds=embeds_clean, num_inference_steps=25, guidance_scale=7.5, generator=generator).images[0]
        
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- СОСТОЯНИЕ №2: Реальный физический коллапс (Дефазировка) ---
    print("[Шаг 2/2] Симуляция декогеренции (Зануление ячеек фазы)...")
    vector_collapsed = base_vector.copy()
    
    # ИСПРАВЛЕНИЕ: Уничтожаем когерентность СТРОГО в фазовом подпространстве
    vector_collapsed[42] = 0.0
    vector_collapsed[43] = 0.0
    
    generator = torch.Generator(device=device).manual_seed(42)
    input_collapsed = torch.tensor(vector_collapsed).unsqueeze(0).to(device)
    with torch.no_grad():
        embeds_collapsed = projector(input_collapsed).half()
        img_collapsed = pipe(prompt_embeds=embeds_collapsed, num_inference_steps=25, guidance_scale=7.5, generator=generator).images[0]

    # Сборка диптиха
    print("\nСборка финального диптиха...")
    width, height = img_clean.size
    diptych_img = Image.new('RGB', (width * 2, height + 50), color=(10, 10, 10))
    
    diptych_img.paste(img_clean, (0, 0))
    diptych_img.paste(img_collapsed, (width, 0))
    
    draw = ImageDraw.Draw(diptych_img)
    draw.text((20, height + 15), "PURE STATE (Coherent Superposition, 45°)", fill=(0, 255, 150))
    draw.text((width + 20, height + 15), "DEPHASED STATE (Quantum Collapse, Phase=0)", fill=(255, 50, 50))
    
    output_path = "results/quantum_collapse_fixed.png"
    diptych_img.save(output_path)
    print(f"[Успех]: Исправленное сравнение сохранено в '{output_path}'!")

if __name__ == "__main__":
    simulate_real_collapse()

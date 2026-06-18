import torch
import torch.nn as nn
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw
import gc

# Архитектура проектора
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

def generate_high_entropy_grid():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Запуск высокоэнтропийного инференса на: {device}")
    
    # Загрузка и оптимизация SD 1.5 под 8GB VRAM
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.enable_attention_slicing()

    # Загрузка проектора
    projector = QuantumToEmbeddingProjector().to(device)
    projector.load_state_dict(torch.load("data/physical_quantum_projector.pt", map_location=device))
    projector.eval()
    
    # Формируем базовый вектор МАКСИМАЛЬНОЙ ЭНТРОПИИ (S_L -> 1)
    np.random.seed(101)
    base_vector = np.zeros(256, dtype=np.float32)
    base_vector[:64] = np.random.normal(0.1, 0.05, 64)   # Подавляем локальный кристаллический порядок
    base_vector[64:] = np.random.normal(0.8, 0.1, 192)  # Доминирование квантового хаоса и запутанности
    
    steps = 5
    phases = np.linspace(0, np.pi / 2, steps)
    generated_images = []
    
    # Фиксируем сид диффузии для чистоты эксперимента
    generator = torch.Generator(device=device).manual_seed(42)
    
    print("\n--- Моделирование квантового хаоса (S_L -> 1) ---")
    for idx, phase in enumerate(phases):
        angle_deg = int(np.degrees(phase))
        print(f"Генерация пены {idx+1}/{steps} | Фаза: {angle_deg}°...")
        
        pauli_vector = base_vector.copy()
        
        # ФИКС ОШИБКИ: Записываем фазу строго в индексы 42 и 43, не уничтожая весь вектор
        pauli_vector[42] = np.cos(phase).astype(np.float32)
        pauli_vector[43] = np.sin(phase).astype(np.float32)
        
        tensor_input = torch.tensor(pauli_vector).unsqueeze(0).to(device)
        with torch.no_grad():
            quantum_prompt_embeds = projector(tensor_input).half()
            
            output = pipe(
                prompt_embeds=quantum_prompt_embeds,
                num_inference_steps=25,
                guidance_scale=7.5,
                generator=generator
            )
            # Извлекаем первое PIL-изображение из списка
            img_result = output.images[0] 
            generated_images.append((img_result, angle_deg))
            
        gc.collect()
        torch.cuda.empty_cache()

    # Сборка высокоэнтропийного коллажа
    print("\nСборка финального хаотического коллажа...")
    # ФИКС СЕТКИ: берем размеры у PIL-картинки внутри первого кортежа
    width, height = generated_images[0][0].size 
    grid_img = Image.new('RGB', (width * steps, height + 40), color=(15, 15, 15))
    
    draw = ImageDraw.Draw(grid_img)
    
    for idx, (img, angle) in enumerate(generated_images):
        grid_img.paste(img, (idx * width, 0))
        text = f"S_L -> 1 | Phase: {angle} deg"
        draw.text((idx * width + 20, height + 10), text, fill=(200, 200, 200))
        
    grid_path = "results/quantum_high_entropy_grid.png"
    grid_img.save(grid_path)
    print(f"[Успех]: Высокоэнтропийная пена сохранена в '{grid_path}'!")

if __name__ == "__main__":
    generate_high_entropy_grid()

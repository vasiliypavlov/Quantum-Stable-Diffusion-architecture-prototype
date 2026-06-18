import torch
import torch.nn as nn
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image
import gc

# 1. Архитектура проектора для корректной десериализации весов
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

def generate_quantum_grid():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Запуск квантового инференса на: {device}")
    
    # 2. Инициализация и жесткая VRAM-оптимизация пайплайна Stable Diffusion 1.5
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, # Экономим VRAM, переходя на FP16
        safety_checker=None        # Отключаем цензор для экономии памяти
    ).to(device)
    
    pipe.enable_attention_slicing() # Оптимизация cross-attention под 8GB VRAM
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[Оптимизация]: xFormers успешно активирован.")
    except Exception:
        print("[Инфо]: xFormers не найден, используется стандартный slicing.")

    # 3. Загрузка обученного квантового инжектора
    projector = QuantumToEmbeddingProjector().to(device)
    projector.load_state_dict(torch.load("data/physical_quantum_projector.pt", map_location=device))
    projector.eval()
    
    # 4. Фиксация базового шума низкой энтропии (чтобы увидеть четкую геометрию)
    np.random.seed(101)
    base_vector = np.zeros(256, dtype=np.float32)
    base_vector[:64] = np.random.normal(1.0, 0.1, 64) # Доминирование локального порядка
    
    # Генерируем 5 фазовых шагов: 0, 22, 45, 67, 90 градусов
    steps = 5
    phases = np.linspace(0, np.pi / 2, steps)
    generated_images = []
    
    # Фиксируем сид диффузии, чтобы менялась только квантовая фаза, а не латентный шум
    generator = torch.Generator(device=device).manual_seed(42)
    
    print("\n--- Начало пошаговой генерации физических макроструктур ---")
    for idx, phase in enumerate(phases):
        angle_deg = int(np.degrees(phase))
        print(f"Генерация кадра {idx+1}/{steps} | Квантовая фаза: {angle_deg}°...")
        
        # Точное кодирование фазового сдвига в Паули-вектор
        pauli_vector = base_vector.copy()
        pauli_vector[42] = np.cos(phase).astype(np.float32)
        pauli_vector[43] = np.sin(phase).astype(np.float32)
        
        # Получаем эмбеддинг через проектор
        tensor_input = torch.tensor(pauli_vector).unsqueeze(0).to(device)
        with torch.no_grad():
            # Переводим в FP16, так как сама SD загружена в полуточности
            quantum_prompt_embeds = projector(tensor_input).half()
            
            # Прямая инжекция эмбеддингов в обход текстового ввода
            output = pipe(
                prompt_embeds=quantum_prompt_embeds,
                num_inference_steps=25, # Оптимально для скорости и качества
                guidance_scale=7.5,
                generator=generator
            )
            image = output.images[0]
            generated_images.append((image, angle_deg))
            
        # Агрессивно очищаем кэш после каждого кадра
        gc.collect()
        torch.cuda.empty_cache()

    # 5. Сборка результатов в горизонтальный Grid
    print("\nСборка финального коллажа...")
    width, height = generated_images[0][0].size
    grid_img = Image.new('RGB', (width * steps, height + 40), color=(30, 30, 30))
    
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(grid_img)
    
    for idx, (img, angle) in enumerate(generated_images):
        grid_img.paste(img, (idx * width, 0))
        # Простая текстовая плашка внизу кадра
        text = f"Phase: {angle} deg"
        draw.text((idx * width + 20, height + 10), text, fill=(255, 255, 255))
        
    grid_path = "results/quantum_phase_evolution_grid.png"
    grid_img.save(grid_path)
    print(f"[Успех]: Эволюция фазы визуализирована и сохранена в '{grid_path}'!")

if __name__ == "__main__":
    generate_quantum_grid()

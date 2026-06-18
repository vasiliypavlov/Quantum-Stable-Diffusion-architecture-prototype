import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# Импортируем архитектуру и декомпозитор из нашего прошлого модуля
from pauli_adapter_sd import PauliBasisDecomposer, QuantumToEmbeddingProjector

def run_quantum_pipeline():
    # Настройки устройства вычислений
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Инициализация сессии генерации QSD. Устройство: {device}")
    
    # 1. Алгебраическая подготовка двух квантовых состояний (из Эксперимента №2)
    decomposer = PauliBasisDecomposer()
    
    test_phi = 1.0 # Фиксированная квантовая фаза
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[15] = np.exp(1j * test_phi) / np.sqrt(2)
    
    # Оригинальное запутанное состояние (содержит фазу когерентности)
    rho_orig = np.outer(psi, np.conj(psi))
    # Дефазированное состояние (матрица принудительно занулена вне диагонали)
    rho_deph = np.diag(np.diag(rho_orig))
    
    # Раскладываем обе матрицы в плоские вещественные 256-мерные векторы Паули
    c_orig = decomposer.decompose(rho_orig)
    c_deph = decomposer.decompose(rho_deph)
    
    # Переводим коэффициенты в тензоры PyTorch
    t_orig = torch.tensor(c_orig, dtype=torch.float32).unsqueeze(0).to(device)
    t_deph = torch.tensor(c_deph, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 2. Создание и инициализация Линейного проектора в [77 x 768]
    projector = QuantumToEmbeddingProjector().to(device)
    projector.eval() # Переводим в режим инференса
    
    with torch.no_grad():
        # Формируем два тензора контекста
        emb_orig = projector(t_orig).to(dtype=torch.float16)
        emb_deph = projector(t_deph).to(dtype=torch.float16)
        
    print(f"Геометрия квантовых эмбеддингов подготовлена: {emb_orig.shape}")

    # 3. Загрузка полносигнальной модели Stable Diffusion 1.5
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    print(f"Загрузка весов {model_id} из Hugging Face Hub...")
    
    # Оптимизируем загрузку для GeForce 3070 (FP16 + снижение фрагментации VRAM)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        safety_checker=None, # Отключаем для экономии VRAM
        requires_safety_checker=False
    ).to(device)
    
    # Включаем пошаговую обработку внимания для удержания лимита в 8 ГБ
    pipe.enable_attention_slicing()
    
    # 4. Процедура инжекции квантового контекста в Cross-Attention
    print("Запуск генерации макроструктур...")
    
    # Фиксируем генератор случайных чисел для жесткого контроля идентичности шума
    latents_seed = 42
    generator = torch.Generator(device=device).manual_seed(latents_seed)
    
    # Генерируем базовый псевдослучайный шум в латентном пространстве
    # Размерность для SD 1.5: [Batch, Channels, Height/8, Width/8] -> [1, 4, 64, 64]
    init_latents = torch.randn((1, 4, 64, 64), generator=generator, device=device, dtype=torch.float16)
    
    # Извлекаем пустые/негативные эмбеддинги для Classifier-Free Guidance (CFG)
    uncond_input = pipe.tokenizer("", padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0].to(dtype=torch.float16)

    # Генерация 1: Из матрицы rho_original
    # Объединяем негативный текстовый эмбеддинг и позитивный квантовый
    context_orig = torch.cat([uncond_embeddings, emb_orig])
    img_orig = pipe(
        prompt_embeds=emb_orig,
        negative_prompt_embeds=uncond_embeddings,
        num_inference_steps=20, # Оптимальное количество шагов для быстрой проверки
        guidance_scale=7.5,
        latents=init_latents
    ).images[0]
    
    # Очищаем кэш видеокарты перед второй тяжелой итерацией
    torch.cuda.empty_cache()
    
    # Генерация 2: Из матрицы rho_dephased (когерентность стёрта)
    img_deph = pipe(
        prompt_embeds=emb_deph,
        negative_prompt_embeds=uncond_embeddings,
        num_inference_steps=20,
        guidance_scale=7.5,
        latents=init_latents
    ).images[0]

    # 5. Сохранение и вывод результатов генерации
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_orig)
    axes[0].set_title("1. QSD Output: ρ_original\n(Квантовая фаза XXXX включена)")
    axes[0].axis('off')
    
    axes[1].imshow(img_deph)
    axes[1].set_title("2. QSD Output: ρ_dephased\n(Только классическая диагональ)")
    axes[1].axis('off')
    
    output_path = os.path.join(output_dir, "quantum_stable_diffusion_output.png")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n[УСПЕХ] Полноразмерные изображения сгенерированы! Паспорт сохранен по пути:\n{output_path}")

if __name__ == "__main__":
    run_quantum_pipeline()

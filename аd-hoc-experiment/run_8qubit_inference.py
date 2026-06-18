import os
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from train_adhoc_projector import AdHocMultiChannelProjector

def run_quantum_inference():
    # На RTX 3070 строго используем CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Внимание: Обнаружен CPU. Для RTX 3070 убедитесь, что установлена PyTorch с поддержкой CUDA!")
    
    print(f"Используем устройство: {device} (GeForce 3070 8GB)")

    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"Загрузка весов Stable Diffusion в режиме FP16...")
    
    # 1. Загружаем модель строго в float16 для экономии VRAM
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    ).to(device)
    
    pipe.safety_checker = None 

    # 2. ВКЛЮЧАЕМ ХАКИ ОПТИМИЗАЦИИ ДЛЯ 8GB VRAM
    print("Включение оптимизаций памяти для RTX 3070...")
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[OK] xFormers успешно активирован")
    except Exception:
        print("[INFO] xFormers не найден, включаем стандартный slicing")
        pipe.enable_attention_slicing() # Альтернатива, если xformers не установлен
        
    pipe.vae.enable_tiling()  # Экономит память на этапе сборки картинки VAE

    # 3. Инициализация нашего 8-кубитного проектора
    # Переводим сам проектор тоже в float16, чтобы не было конфликта типов данных
    projector = AdHocMultiChannelProjector().to(device).to(torch.float16)
    projector.eval()

    # 4. Загрузка квантовых состояний
    try:
        states = np.load("data/states_8q.npy")
        params = np.load("data/params_8q.npy")
    except FileNotFoundError:
        print("Ошибка: Сначала запустите `python generate_8qubit_data.py`!")
        return

    # Берем первый образец
    sample_index = 0
    quantum_state = states[sample_index]
    j_param, g_param = params[sample_index]
    
    print(f"\nПараметры квантовой системы Изинга: J={j_param:.3f}, g={g_param:.3f}")

    # Переводим входной вектор в тензор float16
    quantum_tensor = torch.tensor(quantum_state, dtype=torch.float16).unsqueeze(0).to(device)

    # 5. Проекция квантового состояния в базовый эмбеддинг
    with torch.no_grad():
        quantum_embeddings = projector(quantum_tensor)
        if device.type == "cuda":
            quantum_embeddings = quantum_embeddings.to(torch.float16)

    # --- НОВЫЙ БЛОК: Смешивание с текстовым промптом-руководством ---
    style_prompt = "quantum physics visualization, abstract complex fractal, scientific pattern, high entropy math texture"
    
    # Получаем стандартный эмбеддинг текста из текстового кодировщика модели
    text_inputs = pipe.tokenizer(
        style_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    # Смешиваем: 30% квантового состояния + 70% текстового стиля для баланса
    mixed_embeddings = 0.3 * quantum_embeddings + 0.7 * text_embeddings
    # -----------------------------------------------------------------

    # 6. Безопасная генерация
    print("Генерация абстрактного квантового паттерна...")
    torch.cuda.empty_cache()
    
    image = pipe(
        prompt_embeds=mixed_embeddings, # Передаем смешанный эмбеддинг
        num_inference_steps=30,
        guidance_scale=9.0,             # Немного увеличим силу привязки
        width=512,
        height=512
    ).images[0]

    output_path = f"results/ising_8q_J_{j_param:.2f}_g_{g_param:.2f}.png"
    image.save(output_path)
    print(f"Успех! Изображение сохранено в: {output_path}")

if __name__ == "__main__":
    run_quantum_inference()

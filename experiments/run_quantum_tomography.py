import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image
import os
import numpy as np

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

def run_visual_tomography():
    device = "cpu"
    print(f"Инициализация Эксперимента №9 (Визуальная томография) на: {device}")
    
    # Модели сохраняем в "data", а результаты изображений берем из "results"
    triptych_path = os.path.join("results", "quantum_superposition_triptych.png")
    if not os.path.exists(triptych_path):
        raise FileNotFoundError(f"Критическая ошибка: Файл '{triptych_path}' не найден.")
        
    full_img = Image.open(triptych_path)
    w, h = full_img.size
    single_frame_w = w // 3
    superposition_frame = full_img.crop((single_frame_w, 0, single_frame_w * 2, single_frame_w))
    
    vision_model_id = "openai/clip-vit-base-patch32"
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_id)
    vision_encoder = CLIPVisionModel.from_pretrained(vision_model_id).to(device)
    vision_encoder.eval()
    
    image_inputs = image_processor(images=superposition_frame, return_tensors="pt").to(device)
    with torch.no_grad():
        # Извлекаем глобальный семантический вектор картинки суперпозиции через среднее пулирование [1, 768]
        target_visual_features = vision_encoder(image_inputs.pixel_values).last_hidden_state.mean(dim=1)
        
    projector = MultiChannelQuantumProjector().to(device)
    weights_path = os.path.join("data", "multichannel_quantum_projector.pt")
    projector.load_state_dict(torch.load(weights_path, map_location=device))
    projector.eval()
    
    # Инициализация свободных параметров
    raw_phase = torch.tensor([1.5], requires_grad=True, device=device)
    raw_entropy = torch.tensor([-1.0], requires_grad=True, device=device)
    
    np.random.seed(200)
    base_phase_noise = torch.tensor(np.random.normal(0.5, 0.1, 64).astype(np.float32), device=device)
    base_chaos_noise_low = torch.tensor(np.random.normal(0.1, 0.02, 192).astype(np.float32), device=device)
    base_chaos_noise_high = torch.tensor(np.random.normal(0.8, 0.1, 192).astype(np.float32), device=device)
    
    optimizer = optim.AdamW([raw_phase, raw_entropy], lr=0.1, weight_decay=1e-4)
    
    print("\n--- Сканирование макроструктур кадра (Оптимизация) ---")
    
    iterations = 100
    for step in range(iterations):
        optimizer.zero_grad()
        
        optim_phase = torch.sigmoid(raw_phase) * (np.pi / 2)
        optim_entropy = torch.sigmoid(raw_entropy)
        
        # ФИКС БАГА: Пишем тригонометрию строго по индексам, сохраняя размерность тензора
        phase_vec = base_phase_noise.clone()
        phase_vec[0] = torch.cos(optim_phase)
        phase_vec[1] = torch.sin(optim_phase)
        
        chaos_vec = (1.0 - optim_entropy) * base_chaos_noise_low + optim_entropy * base_chaos_noise_high
        
        # Получаем эмбеддинг и усредняем его по токенам до размерности [1, 768]
        pred_embeddings = projector(phase_vec.unsqueeze(0), chaos_vec.unsqueeze(0)).mean(dim=1)
        
        # Считаем MSE лосс между глобальными семантическими векторами
        loss = nn.functional.mse_loss(pred_embeddings, target_visual_features)
        
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 10 == 0 or step == 0:
            current_angle = int(np.degrees(optim_phase.item()))
            print(f"Шаг {step+1:03d}/{iterations} | Visual Loss: {loss.item():.6f} | Оценка S_L: {optim_entropy.item():.4f} | Оценка Фазы: {current_angle}°")
            
    print("\n--- Финальный отчет квантовой томографии кадра ---")
    final_angle = int(np.degrees(torch.sigmoid(raw_phase).item() * (np.pi / 2)))
    final_entropy = torch.sigmoid(raw_entropy).item()
    
    print(f"Физический профиль системы, извлеченный из изображения:")
    print(f">> Распознанная линейная энтропия S_L: {final_entropy:.4f} (Ожидалось: ~0.50)")
    print(f">> Распознанный угол когерентности: {final_angle}° (Ожидалось: 45°)")
    
    if abs(final_angle - 45) <= 5 and abs(final_entropy - 0.5) <= 0.15:
        print("\n✅ ЭКСПЕРИМЕНТ №9 ЗАВЕРШЕН УСПЕШНО!")
    else:
        print("\n⚠️ ВНИМАНИЕ: Требуется калибровка проекционных матриц.")

if __name__ == "__main__":
    run_visual_tomography()

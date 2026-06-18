import torch
import torch.nn as nn
import torch.optim as optim # ИСПРАВЛЕНИЕ: Добавлен импорт оптимизатора
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image
import os
import numpy as np

# 1. Архитектура промышленного проектора
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

def run_benchmark():
    device = "cpu"
    print("=====================================================", flush=True)
    print("ЗАПУСК АВТОМАТИЧЕСКОГО БЕНЧМАРКА КВАНТОВЫХ МЕТРИК QSD", flush=True)
    print("=====================================================", flush=True)
    
    weights_path = os.path.join("data", "ising_quantum_projector.pt")
    triptych_path = os.path.join("results", "ising_phases_triptych.png")
    
    if not os.path.exists(weights_path) or not os.path.exists(triptych_path):
        raise FileNotFoundError("Критическая ошибка: Отсутствуют веса в 'data/' или триптих в 'results/'!")

    projector = MultiChannelQuantumProjector().to(device)
    projector.load_state_dict(torch.load(weights_path, map_location=device))
    projector.eval()
    
    # -----------------------------------------------------------------
    # ТЕСТ 1: Топологическая непрерывность латентных траекторий
    # -----------------------------------------------------------------
    print("\n[ТЕСТ 1] Анализ гладкости траектории Изинга...")
    np.random.seed(42)
    base_phase = torch.tensor(np.random.normal(0.5, 0.1, 64).astype(np.float32))
    base_chaos = torch.tensor(np.random.normal(0.5, 0.1, 192).astype(np.float32))
    
    steps = 10
    phases = np.linspace(0, np.pi/2, steps)
    embeds = []
    
    with torch.no_grad():
        for p in phases:
            # ФИКС: Тригонометрия вычисляется строго в тензорах без затирания вектора
            p_vec = base_phase.clone()
            p_vec[0] = float(np.cos(p))
            p_vec[1] = float(np.sin(p))
            out = projector(p_vec.unsqueeze(0), base_chaos.unsqueeze(0))
            embeds.append(out.mean(dim=1))
            
    cos = nn.CosineSimilarity(dim=-1)
    similarities = []
    for i in range(1, len(embeds)):
        sim = cos(embeds[i-1], embeds[i]).item()
        similarities.append(sim)
        
    avg_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    print(f"-> Среднее косинусное сходство шагов: {avg_sim:.6f}")
    print(f"-> Минимальное сходство на траектории: {min_sim:.6f}")
    status_t1 = "✅ ПРОЙДЕН" if min_sim > 0.99 else "⚠️ СБОЙ (Обнаружены разрывы)"
    print(f"Статус Теста 1: {status_t1}")

    # -----------------------------------------------------------------
    # ТЕСТ 2: Томографическая биективность визуального кадра фазового перехода
    # -----------------------------------------------------------------
    print("\n[ТЕСТ 2] Проверка томографической биективности по кадру фазового перехода...")
    full_img = Image.open(triptych_path)
    w, h = full_img.size
    single_w = w // 3
    # Вырезаем центральный кадр фазового перехода (S_L = 0.321)
    transition_frame = full_img.crop((single_w, 0, single_w * 2, single_w))
    
    vision_id = "openai/clip-vit-base-patch32"
    processor = CLIPImageProcessor.from_pretrained(vision_id)
    vision_encoder = CLIPVisionModel.from_pretrained(vision_id).to(device)
    vision_encoder.eval()
    
    inputs = processor(images=transition_frame, return_tensors="pt").to(device)
    with torch.no_grad():
        target_features = vision_encoder(inputs.pixel_values).last_hidden_state.mean(dim=1)
        
    # Свободный оптимизируемый параметр энтропии
    raw_entropy = torch.tensor([0.0], requires_grad=True, device=device)
    optimizer = optim.AdamW([raw_entropy], lr=0.2)
    
    base_chaos_low = torch.tensor(np.random.normal(0.1, 0.02, 192).astype(np.float32))
    base_chaos_high = torch.tensor(np.random.normal(0.8, 0.1, 192).astype(np.float32))
    
    for _ in range(50): # Увеличим число итераций до 50 для точной сходимости
        optimizer.zero_grad()
        entropy = torch.sigmoid(raw_entropy)
        chaos_vec = (1.0 - entropy) * base_chaos_low + entropy * base_chaos_high
        pred = projector(base_phase.unsqueeze(0), chaos_vec.unsqueeze(0)).mean(dim=1)
        loss = nn.functional.mse_loss(pred, target_features)
        loss.backward()
        optimizer.step()
        
    final_entropy = torch.sigmoid(raw_entropy).item()
    print(f"-> Истинная энтропия кадра (цель):   0.321")
    print(f"-> Томографическая оценка проектора: {final_entropy:.3f}")
    
    error = abs(final_entropy - 0.321)
    status_t2 = "✅ ПРОЙДЕН" if error <= 0.15 else "⚠️ СБОЙ (Высокая погрешность томографии)"
    print(f"Статус Теста 2: {status_t2}")
    
    print("\n================ BENCHMARK SUMMARY ================")
    if "✅" in status_t1 and "✅" in status_t2:
        print("🎉 ОТЛИЧНО: Промышленный слой Изинга QSD полностью верифицирован!")
    else:
        print("❌ ОШИБКА: Некоторые физические метрики вышли за пределы допусков.")
    print("===================================================", flush=True)

if __name__ == "__main__":
    run_benchmark()

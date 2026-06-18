import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPTextModel, CLIPTokenizer
import os
import time

# Импортируем промышленный загрузчик из нашего конвейера
from data_pipeline import get_ising_dataloader

# 1. Многоканальная архитектура инжектора
class MultiChannelQuantumProjector(nn.Module):
    def __init__(self, phase_dim=64, chaos_dim=192, tokens=77, embed_dim=768):
        super().__init__()
        self.tokens = tokens
        self.embed_dim = embed_dim
        
        # Канал А: Когерентность и геометрическая фаза
        self.phase_encoder = nn.Sequential(
            nn.Linear(phase_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # Канал Б: Энергия термодинамического хаоса и нелокальные корреляции
        self.chaos_encoder = nn.Sequential(
            nn.Linear(chaos_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # Объединенный семантический мост
        self.bridge = nn.Sequential(
            nn.Linear(256 + 256, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(1024, tokens * embed_dim)
        )
        
    def forward(self, phase_vec, chaos_vec):
        feat_phase = self.phase_encoder(phase_vec)
        feat_chaos = self.chaos_encoder(chaos_vec)
        
        # Слияние физических подпространств
        combined_feats = torch.cat((feat_phase, feat_chaos), dim=-1)
        out = self.bridge(combined_feats)
        return out.view(out.size(0), self.tokens, self.embed_dim)

# 2. Функция маскированного подсчета ошибки (Attention-Aware MSE)
def masked_mse_loss(pred_embeds, target_embeds, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand_as(target_embeds)
    
    pred_masked = pred_embeds * mask_expanded
    target_masked = target_embeds * mask_expanded
    
    loss_sum = nn.functional.mse_loss(pred_masked, target_masked, reduction="sum")
    total_active_elements = mask_expanded.sum()
    return loss_sum / (total_active_elements + 1e-8)

# 3. Пайплайн масштабного предобучения
def train_ising_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"===================================================", flush=True)
    print(f"Запуск масштабного предобучения QSD на модели Изинга", flush=True)
    print(f"Устройство вычислений: {device}", flush=True)
    print(f"===================================================", flush=True)
    
    os.makedirs("data", exist_ok=True)
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Инициализация токенизатора и замороженного текстового эксперта CLIP
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
        
    # Инициализация многоканального проектора
    projector = MultiChannelQuantumProjector().to(device)
    
    # Настройка оптимизатора под большой объем данных (50 000 сэмплов)
    optimizer = optim.AdamW(projector.parameters(), lr=3e-4, weight_decay=1e-2)
    
    # 10 эпох на 50к датасете дадут глубокую проработку латентных траекторий
    epochs = 10
    batch_size = 64
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Подготовка конвейера данных
    dataloader = get_ising_dataloader(tokenizer=tokenizer, batch_size=batch_size, num_samples=50000)
    print(f"[Конвейер]: Индустриальный DataLoader готов. Размер батча: {batch_size}.", flush=True)
    
    print("\n--- Старт масштабного итерационного процесса ---", flush=True)
    
    for epoch in range(epochs):
        start_time = time.time()
        projector.train()
        epoch_loss = 0.0
        
        for step, (phase_vec, chaos_vec, input_ids, masks) in enumerate(dataloader):
            phase_vec = phase_vec.to(device)
            chaos_vec = chaos_vec.to(device)
            input_ids = input_ids.to(device)
            masks = masks.to(device)
            
            # Извлекаем эталонные эмбеддинги CLIP
            with torch.no_grad():
                target_embeddings = text_encoder(input_ids).last_hidden_state
                
            optimizer.zero_grad()
            
            # Прямой проход двухканального проектора
            pred_embeddings = projector(phase_vec, chaos_vec)
            
            # Считаем лосс только по значащим физическим концептам
            loss = masked_mse_loss(pred_embeddings, target_embeddings, masks)
            
            loss.backward()
            
            # Защита от взрыва градиентов
            nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            if (step + 1) % 200 == 0:
                print(f"Эпоха [{epoch+1:02d}/{epochs}] | Шаг [{step+1:03d}/{len(dataloader)}] | Текущий Loss: {loss.item():.5f}", flush=True)
                
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        elapsed_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"--- Эпоха {epoch+1:02d} Завершена | Время: {elapsed_time:.1f}s | Masked Loss: {avg_loss:.6f} | LR: {current_lr:.2e} ---", flush=True)
        
    # Сохранение весов промышленной модели в папку data/
    output_path = os.path.join("data", "ising_quantum_projector.pt")
    torch.save(projector.state_dict(), output_path)
    print(f"\n[ГЛОБАЛЬНЫЙ УСПЕХ]: Промышленный проектор Изинга сохранен в '{output_path}'!")

if __name__ == "__main__":
    train_ising_model()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
import os

# Импортируем наш физический датасет из изолированного модуля
from quantum_dataset import PhysicalQuantumCLIPDataset

# 1. Архитектура проектора (QuantumToEmbeddingProjector)
class QuantumToEmbeddingProjector(nn.Module):
    def __init__(self, input_dim=256, tokens=77, embed_dim=768):
        super().__init__()
        self.tokens = tokens
        self.embed_dim = embed_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.05), # Небольшой дропаут для предотвращения коинцидентного переобучения
            
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.05),
            
            nn.Linear(1024, tokens * embed_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        out = self.network(x)
        return out.view(batch_size, self.tokens, self.embed_dim)

# 2. Функция маскированного подсчета среднеквадратичной ошибки (Attention-Aware MSE)
def masked_mse_loss(pred_embeds, target_embeds, attention_mask):
    """
    Вычисляет MSE только для значащих токенов текста, игнорируя PAD-заполнение.
    """
    # Расширяем маску [Batch, 77] до размерности эмбеддингов [Batch, 77, 768]
    mask_expanded = attention_mask.unsqueeze(-1).expand_as(target_embeds)
    
    # Зануляем незначащие токены в обоих тензорах
    pred_masked = pred_embeds * mask_expanded
    target_masked = target_embeds * mask_expanded
    
    # Считаем сумму квадратов разностей
    loss_sum = nn.functional.mse_loss(pred_masked, target_masked, reduction="sum")
    
    # Нормализуем строго по количеству реально существующих токенов в батче
    total_active_elements = mask_expanded.sum()
    return loss_sum / (total_active_elements + 1e-8)

# 3. Основной пайплайн обучения
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Инициализация физического пайплайна на устройстве: {device}")
    
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Загружаем токенизатор и текстовый энкодер CLIP
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    
    # Жестко замораживаем веса CLIP (он выступает в роли экспертного семантического критика)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
        
    # Инициализация проектора и оптимизатора AdamW
    projector = QuantumToEmbeddingProjector().to(device)
    optimizer = optim.AdamW(projector.parameters(), lr=2e-4, weight_decay=1e-2)
    
    # Настройка расписания изменения LR (cosannealing предотвратит застревание на плато)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    
    # Подготовка физического датасета
    num_samples = 4000
    batch_size = 32
    dataset = PhysicalQuantumCLIPDataset(tokenizer=tokenizer, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Датасет успешно сформирован: {num_samples} физических состояний.")
    print("Начало маскированного обучения...")
    
    epochs = 15
    for epoch in range(epochs):
        projector.train()
        epoch_loss = 0.0
        
        for pauli_vecs, input_ids, attention_masks in dataloader:
            # Перенос тензоров на GPU
            pauli_vecs = pauli_vecs.to(device)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            
            # Извлекаем скрытые состояния последней предоконечной выборки CLIP (сквозь "заморозку")
            with torch.no_grad():
                outputs = text_encoder(input_ids)
                target_embeddings = outputs.last_hidden_state # Тензор [Batch, 77, 768]
            
            # Прямой проход нашего квантового инжектора
            optimizer.zero_grad()
            predicted_embeddings = projector(pauli_vecs)
            
            # Расчет ошибки исключительно по значащим текстовым концептам
            loss = masked_mse_loss(predicted_embeddings, target_embeddings, attention_masks)
            
            # Обратное распространение градиентов
            loss.backward()
            
            # Клиппинг градиентов для стабильности обучения на вашей RTX 3070
            nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        # Обновление шага планировщика LR
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Эпоха {epoch+1:02d}/{epochs} | Masked Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
        
    # Сохранение обученной физической модели проектора
    output_filename = "data/physical_quantum_projector.pt"
    torch.save(projector.state_dict(), output_filename)
    print(f"\n[Успех]: Физический квантовый проектор сохранен в '{output_filename}'!")

if __name__ == "__main__":
    main()

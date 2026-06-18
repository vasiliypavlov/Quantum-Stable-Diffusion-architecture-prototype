import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np

# 1. Архитектура проектора (Многослойный перцептрон с нормализацией слоев)
class QuantumToEmbeddingProjector(nn.Module):
    def __init__(self, input_dim=256, tokens=77, embed_dim=768):
        super().__init__()
        self.tokens = tokens
        self.embed_dim = embed_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            
            nn.Linear(1024, tokens * embed_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        out = self.network(x)
        # Возвращаем тензор в формате [Batch, 77, 768]
        return out.view(batch_size, self.tokens, self.embed_dim)

# 2. Датасет, связывающий Паули-векторы с текстовыми концептами
class QuantumCLIPDataset(Dataset):
    def __init__(self, num_samples=2000):
        self.num_samples = num_samples
        
        # Список макрообъектов с разными физическими свойствами для разметки
        self.prompts = [
            # Сепарабельные / Регулярные структуры
            "Perfect crystal lattice, highly ordered, sharp focus, 8k",
            "Monolithic marble cube, clean straight lines, minimalist",
            "Orthogonal grid architecture, perfect symmetry, blueprint style",
            # Запутанные / Диффузные структуры
            "Volumetric smoke cloud, chaotic nebula, deep space photography",
            "Quantum foam, interconnected neural network web, abstract fractal",
            "Turbulent fluid dynamics, splashes of ink in water, macro shot"
        ]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Логика маппинга: первые 3 промпта — регулярные (низкая энтропия),
        # вторые 3 — хаотичные (высокая запутанность)
        prompt_idx = idx % len(self.prompts)
        prompt = self.prompts[prompt_idx]
        
        # Симулируем Паули-вектор (256 вещественных амплитуд)
        # В продакшене здесь должен быть вызов вашего dataset_generator_v2.py
        pauli_vector = np.random.randn(256).astype(np.float32)
        
        # Искусственно зашиваем корреляцию: для регулярных объектов гасим компоненты
        if prompt_idx < 3:
            pauli_vector[128:] *= 0.1 # Подавляем "высокочастотные" нелокальные фазы
        else:
            pauli_vector[:128] *= 0.1 # Подавляем локальный порядок
            
        return torch.tensor(pauli_vector), prompt

# 3. Основной цикл обучения
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")
    
    # Инициализация CLIP
    model_id = "runwayml/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    text_encoder.eval() # Замораживаем CLIP
    
    for param in text_encoder.parameters():
        param.requires_grad = False
        
    # Инициализация нашего проектора
    projector = QuantumToEmbeddingProjector().to(device)
    optimizer = optim.AdamW(projector.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Комбинированный лосс: MSE для точного совпадения + Cosine для геометрии векторов
    mse_loss = nn.MSELoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    
    # Данные
    dataset = QuantumCLIPDataset(num_samples=3000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Обучение
    epochs = 15
    for epoch in range(epochs):
        projector.train()
        epoch_loss = 0
        
        for pauli_vecs, prompts in dataloader:
            pauli_vecs = pauli_vecs.to(device)
            
            # Токенизируем и получаем целевые эмбеддинги от замороженного CLIP
            text_inputs = tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                target_embeddings = text_encoder(text_inputs.input_ids)[0] # [Batch, 77, 768]
                
            # Forward pass проектора
            optimizer.zero_grad()
            predicted_embeddings = projector(pauli_vecs)
            
            # Считаем лосс
            loss_mse = mse_loss(predicted_embeddings, target_embeddings)
            
            # Для Cosine loss выпрямляем тензоры в 2D: [Batch * 77, 768]
            loss_cos = cosine_loss(
                predicted_embeddings.view(-1, 768), 
                target_embeddings.view(-1, 768), 
                torch.ones(predicted_embeddings.size(0) * 77, device=device)
            )
            
            loss = loss_mse + 0.5 * loss_cos
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Эпоха {epoch+1}/{epochs} | Loss: {epoch_loss / len(dataloader):.5f}")
        
    # Сохраняем веса обученного Паули-моста
    torch.save(projector.state_dict(), "data/quantum_to_embedding_projector.pt")
    print("Обучение завершено! Модель сохранена в data/quantum_to_embedding_projector.pt")

if __name__ == "__main__":
    train()

import torch
import numpy as np
import torch.nn as nn

# Повторяем архитектуру для корректной загрузки весов
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

def verify_continuity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Загрузка весов обученного проектора
    projector = QuantumToEmbeddingProjector().to(device)
    projector.load_state_dict(torch.load("data/physical_quantum_projector.pt", map_location=device))
    projector.eval()
    
    # 2. Фиксируем базовое состояние (например, средняя энтропия)
    np.random.seed(42) # Для воспроизводимости базового шума
    base_vector = np.random.normal(0.5, 0.1, 256).astype(np.float32)
    
    # Генерируем траекторию изменения фазы (20 шагов от 0 до pi/2)
    steps = 20
    phases = np.linspace(0, np.pi / 2, steps)
    
    embeddings = []
    
    print("--- Генерация квантовой траектории фазового сдвига ---")
    with torch.no_grad():
        for i, phase in enumerate(phases):
            pauli_vector = base_vector.copy()
            # Внедряем фазу точно так же, как в датасете
            pauli_vector[42] = np.cos(phase).astype(np.float32)
            pauli_vector[43] = np.sin(phase).astype(np.float32)
            
            tensor_input = torch.tensor(pauli_vector).unsqueeze(0).to(device)
            pred_embed = projector(tensor_input) # [1, 77, 768]
            embeddings.append(pred_embed.cpu())
            
    # 3. Анализ метрики непрерывности (Cosine Similarity)
    print("\n--- Анализ гладкости латентного пространства ---")
    print(f"{'Шаг':<5} | {'Угол (град)':<12} | {'Cosine Sim с прошлым шагом':<28} | {'Статус'}")
    print("-" * 65)
    
    cos = nn.CosineSimilarity(dim=-1)
    
    for idx in range(1, len(embeddings)):
        # Берем первый значащий токен эмбеддинга (индекс 1, так как 0 обычно [BOS])
        prev_token = embeddings[idx-1][0, 1, :]
        curr_token = embeddings[idx][0, 1, :]
        
        sim = cos(prev_token.unsqueeze(0), curr_token.unsqueeze(0)).item()
        angle_deg = int(np.degrees(phases[idx]))
        
        # Идеальное латентное пространство должно показывать плавное изменение (sim > 0.95)
        status = "✅ Гладкий переход" if sim > 0.98 else "⚠️ Обнаружен скачок"
        print(f"{idx:02d}    | {angle_deg:<12} | {sim:.6f}                     | {status}")

if __name__ == "__main__":
    verify_continuity()

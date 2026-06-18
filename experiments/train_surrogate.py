import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

# Проверяем и подключаем GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

# 1. Загрузка данных
X = np.load("data/X_matrices.npy")
Y = np.load("data/Y_labels.npy")
E = np.load("data/E_entropies.npy")

# Переводим в тензоры PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.long)
E_tensor = torch.tensor(E, dtype=torch.float32).unsqueeze(1) # Размерность (1000, 1)

dataset = TensorDataset(X_tensor, Y_tensor, E_tensor)

# Разбиваем на Train (80%) и Validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 2. Определение архитектуры сети (Суррогат QSD)
class QSDSurrogateNet(nn.Module):
    def __init__(self):
        super(QSDSurrogateNet, self).__init__()
        
        # Сверточные слои имитируют локальную структуру QCNN
        self.qcnn_features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=2, stride=2), # 16x16 -> 8x8 (локальные гейты)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2), # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten() # 32 * 4 * 4 = 512 признаков
        )
        
        # Общий полносвязный слой-адаптер (прообраз нашего Линейного проектора)
        self.adapter = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        # Голова 1: Классификация (4 класса квантовых сред)
        self.classifier_head = nn.Linear(128, 4)
        
        # Голова 2: Регрессия (метрика запутанности - Линейная энтропия)
        self.entropy_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.qcnn_features(x)
        latent_vector = self.adapter(features)
        
        class_logits = self.classifier_head(latent_vector)
        predicted_entropy = self.entropy_head(latent_vector)
        
        return class_logits, predicted_entropy

# Инициализируем модель и переносим на GeForce 3070
model = QSDSurrogateNet().to(device)

# Функции потерь и оптимизатор
criterion_class = nn.CrossEntropyLoss()
criterion_entropy = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Цикл обучения
epochs = 30
print("\nСтарт обучения суррогатной модели...")

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_y, batch_e in train_loader:
        batch_x, batch_y, batch_e = batch_x.to(device), batch_y.to(device), batch_e.to(device)
        
        optimizer.zero_grad()
        
        # Прямой проход
        logits, pred_entropy = model(batch_x)
        
        # Комбинированная лосс-функция (балансируем классификацию и регрессию)
        loss_c = criterion_class(logits, batch_y)
        loss_e = criterion_entropy(pred_entropy, batch_e)
        loss = loss_c + 10.0 * loss_e # Увеличиваем вес регрессии, так как MSE обычно мал
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_x.size(0)
        
    # Валидация
    model.eval()
    val_correct = 0
    val_entropy_mae = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_e in val_loader:
            batch_x, batch_y, batch_e = batch_x.to(device), batch_y.to(device), batch_e.to(device)
            
            logits, pred_entropy = model(batch_x)
            
            # Считаем точность классификации
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == batch_y).sum().item()
            
            # Считаем среднюю ошибку (MAE) предсказания запутанности
            val_entropy_mae += torch.abs(pred_entropy - batch_e).sum().item()
            
    epoch_loss = train_loss / len(train_dataset)
    val_accuracy = val_correct / len(val_dataset) * 100
    val_mae = val_entropy_mae / len(val_dataset)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Эпоха [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Точность классов: {val_accuracy:.2f}% | Ошибка энтропии (MAE): {val_mae:.4f}")

# 4. Финальный вердикт эксперимента
print("\n--- ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ---")
if val_accuracy > 85.0 and val_mae < 0.05:
    print("📈 [УСПЕХ] Концепт QSD полностью подтвержден на суррогате!")
    print("Модель успешно дифференцирует типы квантовых сред и с высокой точностью считывает уровень запутанности.")
else:
    print("⚠️ [ВНИМАНИЕ] Результаты неоднозначны. Требуется калибровка весов лосс-функции или усложнение QCNN-блока.")

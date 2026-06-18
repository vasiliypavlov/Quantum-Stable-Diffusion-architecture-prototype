import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class AdHocMultiChannelProjector(nn.Module):
    def __init__(self, input_dim=512, seq_len=77, embed_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # Разделение входов на условную макро-фазу и микро-хаос
        self.phase_dim = 128
        self.chaos_dim = input_dim - self.phase_dim
        
        # Проектор канала макро-структуры (низкие частоты)
        self.phase_branch = nn.Sequential(
            nn.Linear(self.phase_dim, 256),
            nn.ReLU(),
            nn.Linear(256, (seq_len * embed_dim) // 2)
        )
        
        # Проектор канала текстур и шума (высокие частоты)
        self.chaos_branch = nn.Sequential(
            nn.Linear(self.chaos_dim, 256),
            nn.ReLU(),
            nn.Linear(256, (seq_len * embed_dim) // 2)
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [batch, 512]
        phase_part = x[:, :self.phase_dim]
        chaos_part = x[:, self.phase_dim:]
        
        out_phase = self.phase_branch(phase_part)
        out_chaos = self.chaos_branch(chaos_part)
        
        # Соединяем каналы воедино
        combined = torch.cat([out_phase, out_chaos], dim=-1)
        # Меняем размерность под требования CLIP/UNet: [batch, 77, 768]
        combined = combined.view(-1, self.seq_len, self.embed_dim)
        return self.layer_norm(combined)

def train():
    # Загрузка сгенерированных данных
    try:
        states = np.load("data/states_8q.npy")
    except FileNotFoundError:
        print("Сначала запустите скрипт генерации данных: generate_8qubit_data.py")
        return

    # Превращаем в тензоры
    X_train = torch.tensor(states, dtype=torch.float32)
    # Создаем фейковые таргеты (в реальном проекте это были бы эмбеддинги CLIP)
    # Имитируем размерность [77, 768]
    Y_train = torch.randn(X_train.shape[0], 77, 768)

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    model = AdHocMultiChannelProjector().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()  # Для "на коленке" берем базовый MSE

    print("Старт быстрого обучения проекта...")
    model.train()
    for epoch in range(1, 11):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Эпоха {epoch}/10 | Лосс: {epoch_loss/len(loader):.6f}")

    print("Экспериментальный проектор успешно обучен и готов к интеграции с UNet!")

if __name__ == "__main__":
    train()

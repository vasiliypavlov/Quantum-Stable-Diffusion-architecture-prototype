import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class QuantumIsingDataset(Dataset):
    def __init__(self, tokenizer, num_samples=50000):
        """
        Промышленный датасет, связывающий 50 000 реальных Паули-векторов Изинга
        с динамической текстовой семантикой для CLIP-выравнивания.
        """
        self.tokenizer = tokenizer
        
        # Загружаем верифицированный бинарный массив 50к х 256
        data_path = os.path.join("data", "quantum_ising_50k.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Датасет не найден по пути {data_path}. Запустите симуляцию!")
            
        self.pauli_vectors = np.load(data_path)[:num_samples]
        
        # Индексы для расчета энтропии на лету (из analyze_quantum_dataset.py)
        self.pauli_indices_A = []
        idx = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        if k == 0 and l == 0:
                            self.pauli_indices_A.append(idx)
                        idx += 1
                        
        # Семантические пулы для генерации физических описаний
        self.order_prompts = [
            "perfect crystal lattice structure", 
            "monolithic geometric grid", 
            "highly ordered molecular array"
        ]
        self.chaos_prompts = [
            "micro-foaming structures, scientific visualization", 
            "non-local correlation wave patterns", 
            "high-frequency turbulent fluid field"
        ]
        self.transition_prompts = [
            "semi-crystalline structural transition", 
            "fluctuating liquid crystal state", 
            "interwoven polymer topology"
        ]
        self.modifiers = ["scientific plot style", "highly detailed", "sharp focus", "8k resolution"]


    def __len__(self):
        return len(self.pauli_vectors)

    def __getitem__(self, idx):
        vector = self.pauli_vectors[idx]
        
        # 1. Извлекаем точную энтропию S_L для текущего состояния
        reduced_coeffs = vector[self.pauli_indices_A]
        tr_rho_A_sq = np.sum(reduced_coeffs ** 2) / 4.0
        s_l = np.clip(1.0 - tr_rho_A_sq, 0.0, 1.0)
        
        # 2. Вычисляем эффективную фазу из первых двух ячеек
        # Извлекаем угол через арктангенс
        phase_angle = np.arctan2(vector[1], vector[0] + 1e-8)
        angle_degrees = int(np.abs(np.degrees(phase_angle))) % 90
        
        # 3. Динамическая семантическая сборка промпта на основе РЕАЛЬНОЙ физики состояния
        if s_l < 0.25:
            core = np.random.choice(self.order_prompts)
            state_descr = f"ordered cold matter at {s_l:.3f} entropy"
        elif s_l > 0.50:
            core = np.random.choice(self.chaos_prompts)
            state_descr = f"correlated thermodynamic chaos at {s_l:.3f} entropy"
        else:
            core = np.random.choice(self.transition_prompts)
            state_descr = f"quantum phase transition state at {s_l:.3f} entropy"
            
        prompt = f"{core}, {state_descr}, rotated at {angle_degrees} degrees diagonal, {np.random.choice(self.modifiers)}"
        
        # 4. Токенизация и генерация масок
        tokens = self.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        
        # Разделяем вектор на два канала для Multi-Channel проектора
        phase_channel = vector[:64]
        chaos_channel = vector[64:]
        
        return (
            torch.tensor(phase_channel, dtype=torch.float32),
            torch.tensor(chaos_channel, dtype=torch.float32),
            tokens.input_ids.squeeze(0),
            tokens.attention_mask.squeeze(0)
        )

def get_ising_dataloader(tokenizer, batch_size=64, num_samples=50000):
    """Инициализатор промышленного даталоадера"""
    dataset = QuantumIsingDataset(tokenizer=tokenizer, num_samples=num_samples)
    # На Windows 11 рекомендуется num_workers=0 для стабильности, либо 2 для параллелизации
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

import numpy as np
import torch
import torch.nn as nn
import os

# 1. Алгебраический генератор базиса Паули для 4 кубитов
class PauliBasisDecomposer:
    def __init__(self):
        # Базовые матрицы Паули (2х2)
        self.I = np.array([[1, 0], [0, 1]], dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        self.matrices = [self.I, self.X, self.Y, self.Z]
        self.basis_names = ['I', 'X', 'Y', 'Z']
        
        # Предвычисляем все 256 тензорных произведений для 4 кубитов
        print("Инициализация 4-кубитного базиса Паули (4^4 = 256 операторов)...")
        self.pauli_256 = []
        self.labels = []
        
        for i, m1 in enumerate(self.matrices):
            for j, m2 in enumerate(self.matrices):
                for k, m3 in enumerate(self.matrices):
                    for l, m4 in enumerate(self.matrices):
                        # Тензорное произведение: m1 ⊗ m2 ⊗ m3 ⊗ m4
                        m12 = np.kron(m1, m2)
                        m123 = np.kron(m12, m3)
                        m1234 = np.kron(m123, m4)
                        
                        self.pauli_256.append(m1234)
                        self.labels.append(f"{self.basis_names[i]}{self.basis_names[j]}{self.basis_names[k]}{self.basis_names[l]}")

    def decompose(self, rho):
        """
        Раскладывает матрицу плотности rho по базису Паули.
        Возвращает 256 вещественных коэффициентов: c_i = Tr(rho * sigma_i)
        """
        coefficients = np.zeros(256, dtype=float)
        for idx, sigma in enumerate(self.pauli_256):
            # c_i = Re(Tr(rho @ sigma)) — мнимая часть всегда равна нулю для эрмитовых матриц
            val = np.trace(rho @ sigma)
            coefficients[idx] = np.real(val)
        return coefficients

# 2. Модуль проектора в контекстное пространство Stable Diffusion [77 x 768]
class QuantumToEmbeddingProjector(nn.Module):
    def __init__(self, input_dim=256, tokens=77, embedding_dim=768):
        super(QuantumToEmbeddingProjector, self).__init__()
        self.tokens = tokens
        self.embedding_dim = embedding_dim
        
        # Проекционный блок: преобразует 256 коэффициентов Паули в 59136 латентных фичей
        # Использование LayerNorm и GeLU стабилизирует инжекцию градиентов
        self.projection_pipeline = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, tokens * embedding_dim),
            nn.LayerNorm(tokens * embedding_dim)
        )

    def forward(self, pauli_coefficients):
        # Ожидаемый вход: [Batch, 256]
        flat_embeddings = self.projection_pipeline(pauli_coefficients)
        # Решейпим в формат контекста Stable Diffusion: [Batch, 77, 768]
        return flat_embeddings.view(-1, self.tokens, self.embedding_dim)

# 3. Демонстрационный запуск и проверка архитектуры
if __name__ == "__main__":
    # Инициализируем декомпозитор базиса
    decomposer = PauliBasisDecomposer()
    
    # Сгенерируем тестовое состояние GHZ с фазой phi=1.0 (из нашего эксперимента №2)
    phi = 1.0
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[15] = np.exp(1j * phi) / np.sqrt(2)
    rho_ghz = np.outer(psi, np.conj(psi))
    
    # Генерируем дефазированную версию (когерентность занулена)
    rho_dephased = np.diag(np.diag(rho_ghz))
    
    # Вычисляем 256 коэффициентов для обоих состояний
    c_ghz = decomposer.decompose(rho_ghz)
    c_deph = decomposer.decompose(rho_dephased)
    
    print("\n--- Физическая проверка разложения Паули ---")
    print(f"Коэффициент при операторах типа 'ZZZZ': {c_ghz[decomposer.labels.index('ZZZZ')]:.4f}")
    # Оператор XXXX должен поймать комплексную фазу когерентности в GHZ
    idx_xxxx = decomposer.labels.index('XXXX')
    print(f"Коэффициент 'XXXX' в исходном GHZ: {c_ghz[idx_xxxx]:.4f}")
    print(f"Коэффициент 'XXXX' после дефазировки: {c_deph[idx_xxxx]:.4f}")
    
    # Разница между состояниями теперь выражена вещественно!
    diff = np.sum(np.abs(c_ghz - c_deph))
    print(f"Суммарная разница векторов Паули (L1-norm): {diff:.4f}")
    
    # Проверка работы PyTorch проектора [77 x 768]
    projector = QuantumToEmbeddingProjector()
    
    # Переводим в тензоры
    t_ghz = torch.tensor(c_ghz, dtype=torch.float32).unsqueeze(0)
    t_deph = torch.tensor(c_deph, dtype=torch.float32).unsqueeze(0)
    
    # Прогоняем через модель
    embedding_ghz = projector(t_ghz)
    embedding_deph = projector(t_deph)
    
    print("\n--- Проверка геометрии тензоров инжекции ---")
    print(f"Размерность выходного эмбеддинга: {embedding_ghz.shape} (должна быть [1, 77, 768])")
    
    if embedding_ghz.shape == (1, 77, 768):
        print("[УСПЕХ] Математический мост Паули -> Stable Diffusion спроектирован корректно!")

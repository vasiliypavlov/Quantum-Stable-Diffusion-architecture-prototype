import numpy as np
import torch
import os
import gc

def get_pauli_matrices():
    """Инициализация базовых матриц Паули 2х2"""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [I, X, Y, Z]

def tensor_product_4site(matrices):
    """Вычисляет тензорное произведение 4 матриц Паули (2x2 -> 16x16)"""
    op = matrices[0]
    for m in matrices[1:]:
        op = np.kron(op, m)
    return op

def generate_pauli_basis():
    """Генерирует полный ортонормированный базис из 256 операторов Паули 16х16"""
    pauli_2x2 = get_pauli_matrices()
    basis_16x16 = []
    
    # 4 в кубе = 256 комбинаций для 4 кубитов
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    ops = [pauli_2x2[i], pauli_2x2[j], pauli_2x2[k], pauli_2x2[l]]
                    basis_16x16.append(tensor_product_4site(ops))
    return basis_16x16

def build_ising_hamiltonian(J, g):
    """Строит гамильтониан модели Изинга 16х16 с периодическими граничными условиями"""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    
    H_interaction = np.zeros((16, 16), dtype=complex)
    H_field = np.zeros((16, 16), dtype=complex)
    
    # Взаимодействие соседних спинов Z_i Z_{i+1}
    H_interaction += np.kron(np.kron(np.kron(Z, Z), I), I) # Z0 Z1
    H_interaction += np.kron(np.kron(np.kron(I, Z), Z), I) # Z1 Z2
    H_interaction += np.kron(np.kron(np.kron(I, I), Z), Z) # Z2 Z3
    H_interaction += np.kron(np.kron(np.kron(Z, I), I), Z) # Z3 Z0 (Периодическая граница)
    
    # Поперечное магнитное поле X_i
    H_field += np.kron(np.kron(np.kron(X, I), I), I)
    H_field += np.kron(np.kron(np.kron(I, X), I), I)
    H_field += np.kron(np.kron(np.kron(I, I), X), I)
    H_field += np.kron(np.kron(np.kron(I, I), I), X)
    
    return -J * H_interaction - g * H_field

def main():
    print("Инициализация генератора квантовых спиновых цепочек Изинга...", flush=True)
    os.makedirs("data", exist_ok=True)
    
    # 1. Генерируем базис Паули для точной декомпозиции
    print("Построение 256-мерного базиса Паули...", flush=True)
    pauli_basis = generate_pauli_basis()
    
    num_samples = 50000
    print(f"Запуск симуляции {num_samples} физических состояний...", flush=True)
    
    # Массивы для хранения результатов
    pauli_vectors = np.zeros((num_samples, 256), dtype=np.float32)
    
    # Генерируем случайную сетку параметров гамильтониана
    np.random.seed(42)
    # Плавное варьирование констант для прохода через критические фазы
    J_values = np.random.uniform(0.1, 2.0, num_samples)
    g_values = np.random.uniform(0.0, 3.0, num_samples)
    temperature_factors = np.random.uniform(0.01, 1.5, num_samples) # Симуляция теплового шума среды
    
    for idx in range(num_samples):
        # Строим гамильтониан спиновой цепочки
        H = build_ising_hamiltonian(J_values[idx], g_values[idx])
        
        # Находим спектр энергий и волновые функции через диагонализацию Эрмитовой матрицы
        energies, states = np.linalg.eigh(H)
        
        # Симулируем термодинамическое смешанное состояние (матрицу плотности rho) при конечной температуре
        # Распределение Больцмана: p_i = exp(-E_i / T)
        beta = 1.0 / (temperature_factors[idx] + 1e-6)
        unnorm_weights = np.exp(-beta * (energies - np.min(energies))) # Сдвиг для численной стабильности
        probabilities = unnorm_weights / np.sum(unnorm_weights)
        
        # Сборка матрицы плотности rho = \sum p_i |psi_i><psi_i|
        rho = np.zeros((16, 16), dtype=complex)
        for i in range(16):
            psi = states[:, i]
            rho += probabilities[i] * np.outer(psi, np.conj(psi))
            
        # Честное разложение матрицы плотности по базису Паули: c_i = Tr(rho * Sigma_i)
        # Так как матрицы Паули эрмитовы, коэффициенты строго вещественные
        for b_idx, sigma in enumerate(pauli_basis):
            pauli_vectors[idx, b_idx] = np.real(np.trace(rho @ sigma))
            
        if (idx + 1) % 10000 == 0:
            print(f"Рассчитано состояний: {idx + 1}/{num_samples}...", flush=True)

    # Сохраняем весь огромный массив Паули-коэффициентов в бинарный сжатый файл
    output_path = os.path.join("data", "quantum_ising_50k.npy")
    np.save(output_path, pauli_vectors)
    print(f"\n[УСПЕХ]: Датасет на 50 000 физических состояний успешно сохранен в '{output_path}'!")
    print(f"Размер файла на диске: ~50 МБ. RAM свободна.")

if __name__ == "__main__":
    main()

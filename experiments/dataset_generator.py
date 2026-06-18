import numpy as np
import scipy.linalg as la
import os
from tqdm import tqdm

# Фиксируем сид для воспроизводимости эксперимента
np.random.seed(42)

def partial_trace_2_2(rho):
    """
    Вычисляет частичный след для системы из 4 кубитов (размер 16х16).
    Трассируем вторую подсистему (кубиты 2, 3), оставляя кубиты 0, 1.
    Размерность подсистем: dim(A) = 4, dim(B) = 4.
    """
    # Решейпим матрицу в тензор (4, 4, 4, 4)
    tensor = rho.reshape((4, 4, 4, 4))
    # Сворачиваем по индексам второй подсистемы (оси 1 и 3 исходного тензора после решейпа)
    # В терминах исходного решейпа: из (A_out, B_out, A_in, B_in) берем след по B
    ptrace = np.einsum('jiki->jk', tensor)
    return ptrace

def get_linear_entropy(rho):
    """Вычисляет линейную энтропию запутанности подсистемы A: S = 1 - Tr(rho_A^2)"""
    rho_A = partial_trace_2_2(rho)
    return float(1.0 - np.real(np.trace(rho_A @ rho_A)))

def generate_random_pure_state():
    """Генерирует случайный вектор чистого состояния |psi> размерности 16"""
    psi = np.random.normal(size=16) + 1j * np.random.normal(size=16)
    return psi / np.linalg.norm(psi)

def generate_class_A():
    """Класс 0: Полностью сепарабельные (незапутанные) чистые состояния"""
    # Произведение 4 случайных одиночных кубитов (dim=2)
    qubits = []
    for _ in range(4):
        q = np.random.normal(size=2) + 1j * np.random.normal(size=2)
        qubits.append(q / np.linalg.norm(q))
    psi = np.kron(np.kron(np.kron(qubits[0], qubits[1]), qubits[2]), qubits[3])
    rho = np.outer(psi, np.conj(psi))
    return rho

def generate_class_B():
    """Класс 1: Максимально запутанные состояния (GHZ-тип с фазовым шумом)"""
    # |GHZ> = (|0000> + e^{i*phi}|1111>) / sqrt(2)
    phi = np.random.uniform(0, 2 * np.pi)
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[15] = np.exp(1j * phi) / np.sqrt(2)
    
    # Добавим небольшое случайное унитарное вращение локальных гейтов для разнообразия
    rho = np.outer(psi, np.conj(psi))
    return rho

def generate_class_C():
    """Класс 2: Частично запутанные состояния с квантовым шумом (дефазировка Phase Flip)"""
    rho = generate_class_B() # Берем запутанное состояние
    p = np.random.uniform(0.2, 0.6) # Сила шума
    
    # Оператор Phase Flip (Z) для одного из кубитов
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    # Применяем Z к первому кубиту: Z X I X I X I
    Z_total = np.kron(np.kron(np.kron(Z, I), I), I)
    
    # Канал дефазировки: (1-p)*rho + p * Z*rho*Z
    rho_noisy = (1 - p) * rho + p * (Z_total @ rho @ Z_total)
    return rho_noisy

def generate_class_D():
    """Класс 3: Максимально смешанные состояния (Тепловой хаос / Деполяризация)"""
    # Берем случайное чистое состояние и сильно топим его в единичной матрице
    psi = generate_random_pure_state()
    rho_pure = np.outer(psi, np.conj(psi))
    p = np.random.uniform(0.7, 0.95) # Высокий уровень шума
    
    identity = np.eye(16, dtype=complex) / 16.0
    rho_mixed = (1 - p) * rho_pure + p * identity
    return rho_mixed

def create_dataset(samples_per_class=250):
    print(f"Генерация квантового датасета ({samples_per_class * 4} состояний)...")
    
    matrices = []
    labels = []
    entropies = []
    
    generators = [generate_class_A, generate_class_B, generate_class_C, generate_class_D]
    
    for class_idx, gen_func in enumerate(generators):
        for _ in tqdm(range(samples_per_class), desc=f"Класс {class_idx}"):
            rho = gen_func()
            entropy = get_linear_entropy(rho)
            
            # Разделяем на реальную и мнимую части для подачи в нейросеть (2 канала, 16x16)
            rho_split = np.stack([np.real(rho), np.imag(rho)], axis=0)
            
            matrices.append(rho_split)
            labels.append(class_idx)
            entropies.append(entropy)
            
    return np.array(matrices), np.array(labels), np.array(entropies)

if __name__ == "__main__":
    X, Y, E = create_dataset(250)
    
    # Сохраняем датасет локально в бинарном формате NumPy
    os.makedirs("data", exist_ok=True)
    np.save("data/X_matrices.npy", X)
    np.save("data/Y_labels.npy", Y)
    np.save("data/E_entropies.npy", E)
    
    print("\n[УСПЕХ] Датасет успешно сгенерирован и сохранен в папку 'data/'!")
    print(f"Формат тензора признаков X: {X.shape} (Образцы, Каналы Re/Im, Высота, Ширина)")
    print(f"Средняя энтропия по датасету: {np.mean(E):.4f}")

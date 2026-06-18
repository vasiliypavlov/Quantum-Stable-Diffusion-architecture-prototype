import os
import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix, kron, eye

# Константы для 8 кубитов
N_QUBITS = 8
DIM = 2**N_QUBITS  # 256

# Операторы Паули
X = csr_matrix([[0, 1], [1, 0]])
Z = csr_matrix([[1, 0], [0, -1]])
I = eye(2)

def get_op(op, site, n_qubits):
    """Строит оператор для конкретного кубита в цепочке"""
    res = eye(1)
    for i in range(n_qubits):
        if i == site:
            res = kron(res, op)
        else:
            res = kron(res, I)
    return csr_matrix(res)

def build_tfim_hamiltonian(J, g):
    """Строит гамильтониан модели Изинга для 8 кубитов с периодическими границами"""
    H = csr_matrix((DIM, DIM))
    # Взаимодействие Z_i * Z_{i+1}
    for i in range(N_QUBITS):
        next_site = (i + 1) % N_QUBITS
        H -= J * (get_op(Z, i, N_QUBITS) @ get_op(Z, next_site, N_QUBITS))
    # Поперечное поле X_i
    for i in range(N_QUBITS):
        H -= g * get_op(X, i, N_QUBITS)
    return H

def generate_dataset(num_samples=100):
    print(f"Генерация {num_samples} квантовых состояний для 8 кубитов...")
    states = []
    parameters = []
    
    for _ in range(num_samples):
        J = np.random.uniform(-2.0, 2.0)
        g = np.random.uniform(0.0, 2.0)
        
        H = build_tfim_hamiltonian(J, g)
        # Находим только минимальное собственное значение (основное состояние)
        eigenvalues, eigenvectors = spla.eigsh(H, k=1, which='SA')
        psi = eigenvectors[:, 0]
        
        # Разворачиваем комплексный вектор [256] в вещественный [512] (чередуем real/imag)
        psi_real_imag = np.stack([psi.real, psi.imag], axis=-1).flatten()
        
        states.append(psi_real_imag)
        parameters.append([J, g])
        
    os.makedirs("data", exist_ok=True)
    np.save("data/states_8q.npy", np.array(states, dtype=np.float32))
    np.save("data/params_8q.npy", np.array(parameters, dtype=np.float32))
    print("Данные успешно сохранены в папку data/")

if __name__ == "__main__":
    generate_dataset(100)

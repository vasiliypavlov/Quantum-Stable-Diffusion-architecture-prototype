import numpy as np
import os
from tqdm import tqdm

np.random.seed(42)

def matrix_to_4channel(rho):
    size = rho.shape[0]
    diag_mask = np.eye(size, dtype=bool)
    off_diag_mask = ~diag_mask
    
    diag_re = np.zeros_like(rho, dtype=float)
    diag_re[diag_mask] = np.real(rho[diag_mask])
    
    off_diag_re = np.zeros_like(rho, dtype=float)
    off_diag_re[off_diag_mask] = np.real(rho[off_diag_mask])
    
    off_diag_im = np.zeros_like(rho, dtype=float)
    off_diag_im[off_diag_mask] = np.imag(rho[off_diag_mask])
    
    structural_mask = np.fromfunction(lambda i, j: np.abs(i - j) / float(size), (size, size), dtype=float)
    return np.stack([diag_re, off_diag_re, off_diag_im, structural_mask], axis=0)

def generate_hard_pair():
    """
    Генерирует пару состояний (сепарабельное и запутанное) 
    с АБСОЛЮТНО идентичной классической диагональю.
    """
    phi = np.random.uniform(0, 2 * np.pi)
    
    # 1. Максимально запутанное состояние |GHZ> с фазой phi
    psi_ent = np.zeros(16, dtype=complex)
    psi_ent[0] = 1.0 / np.sqrt(2)
    psi_ent[15] = np.exp(1j * phi) / np.sqrt(2)
    rho_ent = np.outer(psi_ent, np.conj(psi_ent))
    
    # 2. Сепарабельное (смешанное) состояние с ТОЧНО ТАКОЙ ЖЕ диагональю,
    # но БЕЗ недиагональных элементов (когерентность = 0)
    rho_sep = np.diag(np.diag(rho_ent)).astype(complex)
    
    return rho_sep, rho_ent, phi

def create_unbiased_dataset(samples=500):
    print("Генерация защищенного квантового датасета...")
    matrices = []
    labels = []
    phases = []
    
    for _ in tqdm(range(samples)):
        rho_sep, rho_ent, phi = generate_hard_pair()
        
        # Класс 0: Сепарабельное (Когерентность уничтожена)
        matrices.append(matrix_to_4channel(rho_sep))
        labels.append(0)
        phases.append(0.0) # Фазы нет
        
        # Класс 1: Запутанное (Когерентность сохранена)
        matrices.append(matrix_to_4channel(rho_ent))
        labels.append(1)
        phases.append(phi) # Сохраняем физическую фазу
        
    return np.array(matrices), np.array(labels), np.array(phases)

if __name__ == "__main__":
    X, Y, P = create_unbiased_dataset(500)
    os.makedirs("data", exist_ok=True)
    np.save("data/X_4ch_v2.npy", X)
    np.save("data/Y_labels_v2.npy", Y)
    np.save("data/P_phases_v2.npy", P)
    print(f"[УСПЕХ] Создан датасет. Формат X: {X.shape}")

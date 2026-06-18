import numpy as np
import os

def analyze_dataset():
    print("Запуск экспресс-анализа квантового датасета...", flush=True)
    data_path = os.path.join("data", "quantum_ising_50k.npy")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл {data_path} не найден! Сначала запустите генерацию.")
        
    # Загружаем 50 000 Паули-векторов
    pauli_vectors = np.load(data_path)
    num_samples = pauli_vectors.shape[0]
    print(f"Успешно загружено векторов: {num_samples}. Размерность: {pauli_vectors.shape}", flush=True)
    
    # Инициализируем массив для линейной энтропии подсистемы S_L
    entropy_values = np.zeros(num_samples)
    
    print("Расчет линейной энтропии для 50 000 физических состояний...", flush=True)
    
    # Индексы 16-ти базисных операторов для первой подсистемы из 2-х кубитов (A)
    # В 256-мерном массиве это комбинации, где операторы для кубитов 2 и 3 равны единичной матрице (I)
    # Математически, частичный след Tr_B(rho) эквивалентен фильтрации только этих Паули-коэффициентов!
    pauli_indices_A = []
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    if k == 0 and l == 0:  # Кубиты 2 и 3 находятся в состоянии Identity (I)
                        pauli_indices_A.append(idx)
                    idx += 1
                    
    # Вычисляем энтропию S_L = 1 - Tr(rho_A^2)
    # Для Паули-базиса Tr(rho_A^2) — это просто взвешенная сумма квадратов коэффициентов первой подсистемы
    for idx in range(num_samples):
        vector = pauli_vectors[idx]
        # Извлекаем компоненты подсистемы A и нормируем их (коэффициент 1/4 для 2 кубитов подсистемы)
        reduced_coefficients = vector[pauli_indices_A]
        tr_rho_A_sq = np.sum(reduced_coefficients ** 2) / 4.0
        
        # Линейная энтропия S_L
        entropy_values[idx] = 1.0 - tr_rho_A_sq
        
    # Ограничиваем точность из-за погрешностей плавающей запятой
    entropy_values = np.clip(entropy_values, 0.0, 1.0)
    
    # --- Вывод статистического профиля ---
    print("\n================ STATISTICAL PROFILE ================")
    print(f"Минимальная энтропия (Порядок):       {np.min(entropy_values):.5f}")
    print(f"Максимальная энтропия (Запутанность): {np.max(entropy_values):.5f}")
    print(f"Среднее значение по датасету:        {np.mean(entropy_values):.5f}")
    print(f"Медиана распределения:                {np.median(entropy_values):.5f}")
    print(f"Стандартное отклонение (std):         {np.std(entropy_values):.5f}")
    print("=====================================================")
    
    # Строим текстовую гистограмму распределения для быстрой оценки в консоли
    print("\nТекстовое распределение энтропии (Гистограмма плотности):")
    counts, bins = np.histogram(entropy_values, bins=10, range=(0.0, 1.0))
    for i in range(10):
        bar = "█" * int(counts[i] / (num_samples / 50))  # Масштабируем до 50 символов max
        print(f"[{bins[i]:.1f} - {bins[i+1]:.1f}]: {bar:<50} | {counts[i]} ед.")

if __name__ == "__main__":
    analyze_dataset()

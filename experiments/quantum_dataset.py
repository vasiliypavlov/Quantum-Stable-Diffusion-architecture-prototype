import torch
from torch.utils.data import Dataset
import numpy as np

class PhysicalQuantumCLIPDataset(Dataset):
    def __init__(self, tokenizer, num_samples=5000):
        """
        Датасет для генерации пар [Паули-вектор (256) <-> Текстовый CLIP-эмбеддинг].
        Связывает квантовые инварианты (энтропию и фазу) с макроструктурной семантикой.
        """
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        
        # Семантические пулы для динамической сборки промптов
        self.structures = {
            "low_entropy": [
                "perfect crystal lattice", 
                "monolithic marble cube", 
                "orthogonal geometric grid", 
                "highly ordered architectural blueprint"
            ],
            "mid_entropy": [
                "semi-crystalline structure", 
                "flowing liquid crystal", 
                "interwoven polymer chains", 
                "organized chaos patterns"
            ],
            "high_entropy": [
                "quantum foam", 
                "volumetric smoke nebula", 
                "turbulent fluid dynamics", 
                "chaotic abstract fractal web"
            ]
        }
        self.modifiers = ["sharp focus", "highly detailed", "studio lighting", "8k resolution"]

    def __len__(self):
        return self.num_samples

    def _generate_physical_state(self):
        """Генерирует физические параметры матрицы плотности и кодирует их в Паули-базис"""
        # Линейная энтропия S_L от 0 (порядок) до 1 (максимальная запутанность)
        s_l = np.random.uniform(0.0, 1.0)
        
        # Фазовый угол (наклон / интерференция макроструктур) от 0 до pi/2
        phase = np.random.uniform(0, np.pi / 2)
        
        # Инициализируем вектор Паули-коэффициентов (256 вещественных чисел)
        pauli_vector = np.zeros(256, dtype=np.float32)
        
        # Физический маппинг энтропии S_L:
        if s_l < 0.33:
            # Низкая энтропия -> сильные локальные операторы (первые 64 индекса)
            pauli_vector[:64] = np.random.normal(1.0, 0.2, 64)
            pauli_vector[64:] = np.random.normal(0.0, 0.05, 192)
        elif s_l > 0.66:
            # Высокая энтропия -> доминирование нелокальных многочастичных операторов
            pauli_vector[:64] = np.random.normal(0.1, 0.05, 64)
            pauli_vector[64:] = np.random.normal(0.8, 0.2, 192)
        else:
            # Переходное состояние
            pauli_vector = np.random.normal(0.5, 0.1, 256).astype(np.float32)
            
        # Жестко зашиваем фазу в индексы 42 и 43 (эмуляция коллапса когерентности из Эксп. №3)
        pauli_vector[42] = np.cos(phase).astype(np.float32)
        pauli_vector[43] = np.sin(phase).astype(np.float32)
        
        return pauli_vector, s_l, phase

    def __getitem__(self, idx):
        pauli_vector, s_l, phase = self._generate_physical_state()
        
        # Динамическая сборка текстового описания на основе квантовой физики
        if s_l < 0.33:
            core_prompt = np.random.choice(self.structures["low_entropy"])
            entropy_descr = "perfectly ordered cold matter"
        elif s_l > 0.66:
            core_prompt = np.random.choice(self.structures["high_entropy"])
            entropy_descr = "maximally entangled thermodynamic chaos"
        else:
            core_prompt = np.random.choice(self.structures["mid_entropy"])
            entropy_descr = "transition state of matter"
            
        # Внедряем фазовый модификатор геометрии (угол поворота)
        angle_degrees = int(np.degrees(phase))
        phase_descr = f"rotated at {angle_degrees} degrees diagonal"
        
        # Финальный промпт для выравнивания с CLIP
        modifier = np.random.choice(self.modifiers)
        prompt = f"{core_prompt}, {entropy_descr}, {phase_descr}, {modifier}"
        
        # Токенизация для выделения значащих токенов и маскирования PAD-отступов
        tokens = self.tokenizer(
            prompt, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        )
        
        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)
        
        return torch.tensor(pauli_vector), input_ids, attention_mask

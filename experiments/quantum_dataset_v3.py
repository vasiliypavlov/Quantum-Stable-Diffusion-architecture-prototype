import torch
from torch.utils.data import Dataset
import numpy as np

class MultiChannelQuantumDataset(Dataset):
    def __init__(self, tokenizer, num_samples=4000):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        
        self.structures = {
            "low_entropy": ["perfect crystal lattice", "monolithic marble cube", "orthogonal geometric grid"],
            "mid_entropy": ["semi-crystalline structure", "flowing liquid crystal", "interwoven polymer chains"],
            "high_entropy": ["quantum foam", "volumetric smoke nebula", "turbulent fluid dynamics"]
        }
        self.modifiers = ["sharp focus", "highly detailed", "8k resolution"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        s_l = np.random.uniform(0.0, 1.0)
        phase = np.random.uniform(0, np.pi / 2)
        
        # Создаем два изолированных физических канала
        phase_channel = np.zeros(64, dtype=np.float32)
        chaos_channel = np.zeros(192, dtype=np.float32)
        
        # Заполняем каналы в зависимости от энтропии
        if s_l < 0.33:
            phase_channel[:] = np.random.normal(1.0, 0.2, 64)
            chaos_channel[:] = np.random.normal(0.0, 0.05, 192)
            core_prompt = np.random.choice(self.structures["low_entropy"])
            entropy_descr = "perfectly ordered cold matter"
        elif s_l > 0.66:
            phase_channel[:] = np.random.normal(0.1, 0.05, 64)
            chaos_channel[:] = np.random.normal(0.8, 0.2, 192)
            core_prompt = np.random.choice(self.structures["high_entropy"])
            entropy_descr = "maximally entangled thermodynamic chaos"
        else:
            phase_channel[:] = np.random.normal(0.5, 0.1, 64)
            chaos_channel[:] = np.random.normal(0.5, 0.1, 192)
            core_prompt = np.random.choice(self.structures["mid_entropy"])
            entropy_descr = "transition state of matter"
            
        # Зашиваем фазу в фазовый канал (индексы 42 и 43 внутри блока 0:64)
        phase_channel[42] = np.cos(phase).astype(np.float32)
        phase_channel[43] = np.sin(phase).astype(np.float32)
        
        angle_degrees = int(np.degrees(phase))
        prompt = f"{core_prompt}, {entropy_descr}, rotated at {angle_degrees} degrees diagonal, {np.random.choice(self.modifiers)}"
        
        tokens = self.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        
        return (
            torch.tensor(phase_channel), 
            torch.tensor(chaos_channel), 
            tokens.input_ids.squeeze(0), 
            tokens.attention_mask.squeeze(0)
        )

import torch
import torch.nn as nn
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw
import gc

class MultiChannelQuantumProjector(nn.Module):
    def __init__(self, phase_dim=64, chaos_dim=192, tokens=77, embed_dim=768):
        super().__init__()
        self.tokens = tokens
        self.embed_dim = embed_dim
        self.phase_encoder = nn.Sequential(nn.Linear(phase_dim, 256), nn.LayerNorm(256), nn.GELU())
        self.chaos_encoder = nn.Sequential(nn.Linear(chaos_dim, 256), nn.LayerNorm(256), nn.GELU())
        self.bridge = nn.Sequential(nn.Linear(256 + 256, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.05), nn.Linear(1024, tokens * embed_dim))
    def forward(self, phase_vec, chaos_vec):
        f_p = self.phase_encoder(phase_vec)
        f_c = self.chaos_encoder(chaos_vec)
        return self.bridge(torch.cat((f_p, f_c), dim=-1)).view(-1, self.tokens, self.embed_dim)

def test_multichannel_collapse():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.enable_attention_slicing()

    projector = MultiChannelQuantumProjector().to(device)
    projector.load_state_dict(torch.load("data/multichannel_quantum_projector.pt", map_location=device))
    projector.eval()
    
    # Фиксируем высокоэнтропийный базовый хаос (S_L -> 1)
    np.random.seed(101)
    base_phase_channel = np.random.normal(0.1, 0.05, 64).astype(np.float32)
    base_chaos_channel = np.random.normal(0.8, 0.1, 192).astype(np.float32)
    
        # --- 1. Чистая суперпозиция (45 градусов) ---
    p_clean = base_phase_channel.copy()
    p_clean[42] = np.cos(np.pi / 4).astype(np.float32)
    p_clean[43] = np.sin(np.pi / 4).astype(np.float32)
    
    gen = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        emb_clean = projector(torch.tensor(p_clean).unsqueeze(0).to(device), torch.tensor(base_chaos_channel).unsqueeze(0).to(device)).half()
        output_clean = pipe(prompt_embeds=emb_clean, num_inference_steps=25, guidance_scale=7.5, generator=gen)
        img_clean = output_clean.images[0] # ИСПРАВЛЕНИЕ: достаем PIL-изображение из списка
        
    # --- 2. Физический коллапс (Зануление фазы) ---
    p_collapsed = base_phase_channel.copy()
    p_collapsed[42] = 0.0
    p_collapsed[43] = 0.0
    
    gen = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        emb_collapsed = projector(torch.tensor(p_collapsed).unsqueeze(0).to(device), torch.tensor(base_chaos_channel).unsqueeze(0).to(device)).half()
        output_collapsed = pipe(prompt_embeds=emb_collapsed, num_inference_steps=25, guidance_scale=7.5, generator=gen)
        img_collapsed = output_collapsed.images[0] # ИСПРАВЛЕНИЕ: достаем PIL-изображение из списка

    # Сборка диптиха
    width, height = img_clean.size # Теперь .size отработает корректно!
    diptych = Image.new('RGB', (width * 2, height + 50), color=(10, 10, 10))
    diptych.paste(img_clean, (0, 0))
    diptych.paste(img_collapsed, (width, 0))
    
    draw = ImageDraw.Draw(diptych)
    draw.text((20, height + 15), "PURE STATE (Coherent 45° Chaos)", fill=(0, 255, 150))
    draw.text((width + 20, height + 15), "DEPHASED STATE (Phase Collapsed)", fill=(255, 50, 50))
    diptych.save("results/multichannel_collapse_test.png")
    print("[Успех]: Новый тест сохранен в 'multichannel_collapse_test.png'")

if __name__ == "__main__":
    test_multichannel_collapse()

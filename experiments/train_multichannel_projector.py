import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from quantum_dataset_v3 import MultiChannelQuantumDataset

class MultiChannelQuantumProjector(nn.Module):
    def __init__(self, phase_dim=64, chaos_dim=192, tokens=77, embed_dim=768):
        super().__init__()
        self.tokens = tokens
        self.embed_dim = embed_dim
        
        # Канал А: Обработка геометрии и фазы
        self.phase_encoder = nn.Sequential(
            nn.Linear(phase_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # Канал Б: Обработка энергии хаоса
        self.chaos_encoder = nn.Sequential(
            nn.Linear(chaos_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        # Объединенный мост семантического выравнивания
        self.bridge = nn.Sequential(
            nn.Linear(256 + 256, 1024), # Соединяем фичи обеих голов
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(1024, tokens * embed_dim)
        )
        
    def forward(self, phase_vec, chaos_vec):
        feat_phase = self.phase_encoder(phase_vec)
        feat_chaos = self.chaos_encoder(chaos_vec)
        
        # Конкатенируем скрытые пространства
        combined_feats = torch.cat((feat_phase, feat_chaos), dim=-1)
        out = self.bridge(combined_feats)
        return out.view(out.size(0), self.tokens, self.embed_dim)

def masked_mse_loss(pred_embeds, target_embeds, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand_as(target_embeds)
    loss_sum = nn.functional.mse_loss(pred_embeds * mask_expanded, target_embeds * mask_expanded, reduction="sum")
    return loss_sum / (mask_expanded.sum() + 1e-8)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-v1-5"
    
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    text_encoder.eval()
    for param in text_encoder.parameters(): param.requires_grad = False
        
    projector = MultiChannelQuantumProjector().to(device)
    optimizer = optim.AdamW(projector.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
    
    dataset = MultiChannelQuantumDataset(tokenizer=tokenizer, num_samples=4000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("Запуск многоканального принудительного обучения...")
    for epoch in range(12):
        projector.train()
        epoch_loss = 0.0
        for phase_vec, chaos_vec, input_ids, masks in dataloader:
            phase_vec, chaos_vec = phase_vec.to(device), chaos_vec.to(device)
            input_ids, masks = input_ids.to(device), masks.to(device)
            
            with torch.no_grad():
                target_embeddings = text_encoder(input_ids).last_hidden_state
                
            optimizer.zero_grad()
            pred_embeddings = projector(phase_vec, chaos_vec)
            loss = masked_mse_loss(pred_embeddings, target_embeddings, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        print(f"Эпоха {epoch+1:02d}/12 | Loss: {epoch_loss/len(dataloader):.6f}")
        
    torch.save(projector.state_dict(), "data/multichannel_quantum_projector.pt")
    print("Модель успешно сохранена в data/multichannel_quantum_projector.pt")

if __name__ == "__main__":
    main()

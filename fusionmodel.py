import os
import pandas as pd
import torch
import torchvision.io as io
from sklearn.metrics import accuracy_score
from transformers import logging
logging.set_verbosity_error()
import torchaudio.transforms as T
from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_embeddings", exist_ok=True)

LATENT_DIM       = 256    
MODALITY_DROPOUT = 0.2     
NUM_CLASSES      = 2
MAX_EPOCHS       = 15
LR_MODEL         = 1e-4
LR_MAX           = 1e-3
OUTPUT_DIR       = "saved_models"


class ModalityAttentionFusionModel(nn.Module):
    def __init__(self, input_dims, attn_dim=128, hidden_dim=512, num_classes=NUM_CLASSES):
        super().__init__()
        self.n_mod   = len(input_dims)
        raw_dim      = input_dims[0]
        assert len(set(input_dims)) == 1, "All embeddings must share the same raw dim"

        # projection to latent
        self.proj_layers = nn.ModuleList([
            nn.Linear(raw_dim, LATENT_DIM) for _ in range(self.n_mod)
        ])
        # per-modality batchnorm
        self.bn_layers   = nn.ModuleList([
            nn.BatchNorm1d(LATENT_DIM) for _ in range(self.n_mod)
        ])
        # per-modality gating
        self.gate_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(LATENT_DIM,1), nn.Sigmoid())
            for _ in range(self.n_mod)
        ])

        # attention projection per modality (latent → attn_dim)
        self.attn_projs = nn.ModuleList([
            nn.Linear(LATENT_DIM, attn_dim) for _ in range(self.n_mod)
        ])
        self.attn_vec = nn.Parameter(torch.randn(attn_dim))

        # classifier on fused latent
        self.classifier = nn.Sequential(
            nn.Linear(LATENT_DIM, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x, return_embedding=False, return_attn=False):
      B, total = x.size()
      raw_dim = total // self.n_mod
      chunks  = x.view(B, self.n_mod, raw_dim)

      z_chunks = []
      for i in range(self.n_mod):
          z = self.proj_layers[i](chunks[:, i])
          z = self.bn_layers[i](z)
          g = self.gate_layers[i](z)
          z_chunks.append(z * g)
      z = torch.stack(z_chunks, dim=1)  

      # modality dropout
      if self.training:
          mask = (torch.rand(B, self.n_mod, device=x.device) > MODALITY_DROPOUT)\
                .float().unsqueeze(-1)
          z = z * mask

      # attention scores
      scores = []
      for i, proj in enumerate(self.attn_projs):
          h = torch.tanh(proj(z[:, i]))      
          score = (h * self.attn_vec).sum(dim=1)  
          scores.append(score)
      scores = torch.stack(scores, dim=1)       

      weights = torch.softmax(scores, dim=1)     
      fused   = (weights.unsqueeze(-1) * z).sum(dim=1)  

      if return_attn:
          return self.classifier(fused), weights
      elif return_embedding:
          return fused
      else:
          return self.classifier(fused)

def load_embeddings(modality_name, split):
    path = f"/content/drive/MyDrive/multimodal_empathy/models/45/{modality_name}/{modality_name}_{split}_emb.pt"
    data = torch.load(path)
    return data['embeddings'], data['labels']


def build_fusion_dataset(modalities, split):
    embs_list, labels = [], None
    for m in modalities:
        embs, lbls = load_embeddings(m, split)  
        embs_list.append(embs)
        labels = lbls if labels is None else labels

    fused = torch.cat(embs_list, dim=1) 
    dims  = [embs.size(1) for embs in embs_list]
    return TensorDataset(fused, labels), dims


def train_fusion_model(model, train_loader, val_loader, max_epochs=MAX_EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=LR_MODEL, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX,
        steps_per_epoch=len(train_loader), epochs=max_epochs
    )
    criterion = nn.CrossEntropyLoss()
    best_val  = 0.0

    for _ in range(max_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        val_acc = evaluate_model(model, val_loader)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "saved_models/fusion_best.pth")

    model.load_state_dict(torch.load("saved_models/fusion_best.pth"))
    return model


def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total
def evaluate_with_attention(model, loader):
    model.eval()
    correct, total = 0, 0
    all_attn_weights = []
    with torch.no_grad():
        for x, y in loader:
            logits, attn_weights = model(x, return_attn=True)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)
            all_attn_weights.append(attn_weights.cpu())

    acc = correct / total
    all_attn_weights = torch.cat(all_attn_weights, dim=0) 

    return acc, all_attn_weights.numpy()

def run_multimodal_experiment(modalities, batch_size=64):
    train_ds, dims = build_fusion_dataset(modalities, 'train')
    val_ds,   _    = build_fusion_dataset(modalities, 'val')
    test_ds,  _    = build_fusion_dataset(modalities, 'test')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = ModalityAttentionFusionModel(dims, attn_dim=128, hidden_dim=512)

    model = train_fusion_model(model, train_loader, val_loader)

    acc = evaluate_model(model, test_loader)

    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"{modalities} → Acc: {acc:.4f}, F1: {f1:.4f}")
    return acc, f1, model, test_loader

text_model, audio_model, video_model = "roberta", "hubert", "videomae"

_, _, text_m, _ = run_multimodal_experiment([text_model])
_, _, audio_m, _ = run_multimodal_experiment([audio_model])
_, _, video_m, _ = run_multimodal_experiment([video_model])

run_multimodal_experiment([text_model, audio_model])
run_multimodal_experiment([text_model, video_model])
run_multimodal_experiment([audio_model, video_model])
_, _, full_m, test_data = run_multimodal_experiment([text_model, audio_model, video_model])
torch.save(full_m.state_dict(), os.path.join(OUTPUT_DIR, f"fullmodel_best.pth"))

def compute_unimodal_vs_full_disagreements(models_dict, model_names, full_model, full_modalities, batch_size=64):
    from torch.utils.data import DataLoader
    full_ds, _ = build_fusion_dataset(full_modalities, 'test')
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=False)
    full_model.eval()
    full_preds = []
    with torch.no_grad():
        for x, _ in full_loader:
            logits = full_model(x)
            full_preds.append(logits.argmax(dim=1).cpu())
    full_preds = torch.cat(full_preds)

    disagreement_rates = {}

    for modality in model_names:
        uni_model = models_dict[modality]
        uni_ds, _ = build_fusion_dataset([modality], 'test')
        uni_loader = DataLoader(uni_ds, batch_size=batch_size, shuffle=False)

        uni_model.eval()
        uni_preds = []
        with torch.no_grad():
            for x, _ in uni_loader:
                logits = uni_model(x)
                uni_preds.append(logits.argmax(dim=1).cpu())
        uni_preds = torch.cat(uni_preds)

        disagreement = (uni_preds != full_preds).float().mean().item()
        disagreement_rates[modality] = disagreement
        print(f"Disagreement ({modality} vs full): {disagreement:.3f}")

    return disagreement_rates
models = {
    "roberta": text_m,
    "hubert": audio_m,
    "videomae": video_m
}

disagreements = compute_unimodal_vs_full_disagreements(
    models_dict=models,
    model_names=["roberta", "hubert", "videomae"],
    full_model=full_m,
    full_modalities=["roberta", "hubert", "videomae"]
)
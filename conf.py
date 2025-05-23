import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from fusionmodel import text_m, audio_m, video_m, full_m
import numpy as np 

full_key = "full"
splits = ["test","val"]
def load_with_metadata(modality, split):
    data = torch.load(f"/content/drive/MyDrive/multimodal_empathy/models/45/{modality}/{modality}_{split}_emb.pt")
    emb = data["embeddings"]
    labels = data["labels"]
    meta = pd.DataFrame(data["metadata"]) 
    return emb, labels, meta
def get_confidences(model, X, y, batch_size=64):
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    confs = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            probs = F.softmax(model(xb), dim=1)
            conf = probs.gather(1, yb.view(-1, 1)).squeeze(1)
            confs.append(conf.cpu().numpy())
    return np.concatenate(confs)
mods = ["roberta", "hubert", "videomae"]
embs, labels, metas = {s: {} for s in splits}, {s: {} for s in splits}, {s: {} for s in splits}
for split in splits:
    for m in mods:
        embs[split][m], labels[split][m], metas[split][m] = load_with_metadata(m, split=split)
    embs[split][full_key] = torch.cat([embs[split][m] for m in mods], dim=1)
    labels[split][full_key] = labels[split]["roberta"]
    metas[split][full_key] = metas[split]["roberta"]

conf = {s: {} for s in splits}
conf["test"]["text"]  = get_confidences(text_m,  embs["test"]["roberta"],   labels["test"]["roberta"])
conf["test"]["audio"] = get_confidences(audio_m, embs["test"]["hubert"],    labels["test"]["hubert"])
conf["test"]["video"] = get_confidences(video_m, embs["test"]["videomae"],  labels["test"]["videomae"])
conf["test"]["full"]  = get_confidences(full_m,  embs["test"][full_key],    labels["test"][full_key])

conf["val"]["text"]  = get_confidences(text_m,  embs["val"]["roberta"],   labels["val"]["roberta"])
conf["val"]["audio"] = get_confidences(audio_m, embs["val"]["hubert"],    labels["val"]["hubert"])
conf["val"]["video"] = get_confidences(video_m, embs["val"]["videomae"],  labels["val"]["videomae"])
conf["val"]["full"]  = get_confidences(full_m,  embs["val"][full_key],    labels["val"][full_key])
pairs = [('text','full'), ('audio','full'), ('video','full')]
confidences = {
    "text": np.concatenate([conf["test"]["text"], conf["val"]["text"]]),
    "audio": np.concatenate([conf["test"]["audio"], conf["val"]["audio"]]),
    "video": np.concatenate([conf["test"]["video"], conf["val"]["video"]]),
    "full": np.concatenate([conf["test"]["full"],  conf["val"]["full"]])
}
fig = plt.figure(figsize=(6, 16))

gs = GridSpec(nrows=6, ncols=2,
              height_ratios=[0.5, 4,   0.5, 4,   0.5, 4],
              width_ratios =[4,   1],
              hspace=0.3, wspace=0.1)

for i, (mod, full) in enumerate(pairs):
    hist_row  = 2*i
    scat_row  = 2*i + 1

    ax_histx = fig.add_subplot(gs[hist_row, 0])     
    ax_scatter = fig.add_subplot(gs[scat_row, 0], sharex=ax_histx) 
    ax_histy = fig.add_subplot(gs[scat_row, 1], sharey=ax_scatter)

    x = confidences[mod]
    y = confidences[full]

    ax_scatter.scatter(x, y, alpha=0.5, s=15)
    ax_scatter.plot([0,1],[0,1],'r--')
    ax_scatter.set_xlim(0,1)
    ax_scatter.set_ylim(0,1)

    ax_scatter.set_xlabel(f"{mod.capitalize()} Confidence")
    if i == 0:
        ax_scatter.set_ylabel("Full Model Confidence")
    else:
        ax_scatter.set_yticklabels([])

    ax_histx.hist(x, bins=20)
    ax_histx.axis('off')
    ax_histy.hist(y, bins=20, orientation='horizontal')
    ax_histy.axis('off')

plt.tight_layout(rect=[0,0,1,0.95])
plt.show()
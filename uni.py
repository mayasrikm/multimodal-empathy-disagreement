import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoModelForSequenceClassification,
    AutoModelForAudioClassification,
    AutoModelForVideoClassification
)
import torchvision.io as io
import difflib
from sklearn.metrics import accuracy_score
from transformers import logging
logging.set_verbosity_error()
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoModelForSequenceClassification,
    AutoModelForVideoClassification,
   AutoModelForAudioClassification
)
import av
import random
import numpy as np

MODEL_SPECS = [
    {"name": "hubert",      "checkpoint": "facebook/hubert-base-ls960", "modality": "audio"},
    {"name": "roberta",     "checkpoint": "roberta-base","modality": "text"},
    {"name": "deberta",     "checkpoint": "microsoft/deberta-v3-base",   "modality": "text"},
    {"name": "wav2vec2",       "checkpoint": "facebook/wav2vec2-base",     "modality": "audio"},
    {"name": "videomae",    "checkpoint": "MCG-NJU/videomae-base",   "modality": "video"},
    {"name": "timesformer", "checkpoint": "facebook/timesformer-base-finetuned-k400","modality": "video"},
]

DATA_DIR    = "/content/drive/MyDrive/multimodal_empathy"
AUDIO_DIR   = os.path.join(DATA_DIR, "Audios")
VIDEO_DIR   = os.path.join(DATA_DIR, "videos")
OUTPUT_DIR  = "./outputs_5e-6"
BATCH_SIZE  = 8
NUM_EPOCHS  = 15
LR          = 5e-6
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)
df_train = pd.read_csv(os.path.join(DATA_DIR, "dataset/train.csv"))
df_val   = pd.read_csv(os.path.join(DATA_DIR, "dataset/val.csv"))
df_test  = pd.read_csv(os.path.join(DATA_DIR, "dataset/test.csv"))
unique_labels = df_train["label"].unique().tolist()
label2id = {lab: i for i, lab in enumerate(unique_labels)}
id2label = {i: lab for lab, i in label2id.items()}



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, label2id):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        enc = self.tok(
            row["transcript"],
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc.input_ids[0],
            "attention_mask": enc.attention_mask[0],
            "labels":         torch.tensor(self.label2id[row["label"]], dtype=torch.long)
        }

def collate_text(batch):
    input_ids      = [b["input_ids"]      for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels         = torch.stack([b["labels"] for b in batch])

    input_ids      = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class AudioDataset(Dataset):
    def __init__(self, df, audio_dir, feat_extr, label2id):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.fe = feat_extr
        self.target_sr = feat_extr.sampling_rate
        self.resampler = None
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        path = os.path.join(self.audio_dir, row["filename"] + ".wav")
        wav, sr = torchaudio.load(path)
        if sr != self.target_sr:
            if (self.resampler is None) or (self.resampler.orig_freq != sr):
                self.resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            wav = self.resampler(wav)
        max_len_samples = int(self.target_sr * 10)
        if wav.shape[1] > max_len_samples:
            wav = wav[:, :max_len_samples]

        wav = wav.squeeze(0)                      
        wav = wav / max(wav.abs().max(), 1e-6)     
        wav = wav.numpy().astype("float32")       
        enc = self.fe(wav, sampling_rate=self.target_sr, return_tensors="pt", padding=False)
        attention_mask = torch.ones_like(enc.input_values[0], dtype=torch.int64)

        return {
            "input_values":   enc.input_values[0],
            "attention_mask": attention_mask,
            "labels":         torch.tensor(self.label2id[row["label"]], dtype=torch.long)
        }

def collate_audio(batch):
    ivs   = [b["input_values"]   for b in batch]
    masks = [b["attention_mask"] for b in batch]
    labels= torch.stack([b["labels"] for b in batch])
    ivs   = torch.nn.utils.rnn.pad_sequence(ivs, batch_first=True, padding_value=0)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    return {"input_values": ivs, "attention_mask": masks, "labels": labels}



def load_my_video(path, num_frames=16, target_size=(224,224)):
    """
    Reads a video file and returns `num_frames` evenly‐spaced frames as a
    tensor of shape (num_frames, C, H, W), normalized to [0,1].
    Pads or truncates as needed.
    """
    video, _, info = io.read_video(path, pts_unit="sec")
    if video.numel() == 0 or video.shape[0] == 0:
        raise ValueError(f"No frames found in video: {path}")

 
    T, H, W, C = video.shape
    fps = info["video_fps"]
    max_frames = int(10 * fps)
    if T > max_frames:
        video = video[:max_frames]

    if video.shape[0] < num_frames:
        pad_count = num_frames - video.shape[0]
        last_frame = video[-1:].repeat(pad_count, 1, 1, 1) 
        video = torch.cat([video, last_frame], dim=0)

    idxs = torch.linspace(0, video.shape[0] - 1, steps=num_frames).long()
    frames = video[idxs]                       

    frames = frames.permute(0, 3, 1, 2).float() / 255.0
    return frames


class VideoDataset(Dataset):
    def __init__(self, df, video_dir, feat_extr, label2id):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.fe = feat_extr
        self.label2id = label2id
        self.files = [
          f for f in os.listdir(video_dir)
          if f.lower().endswith(".mp4")
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
      row  = self.df.loc[idx]
      stem = os.path.splitext(row["filename"])[0]
      candidate = stem + ".mp4"
      if candidate in self.files:
          video_path = os.path.join(self.video_dir, candidate)
      else:
          yt_id = stem.split("_", 1)[0]
          cands = [f for f in self.files if f.startswith(yt_id)]
          if cands:
              video_path = os.path.join(self.video_dir, cands[0])
          else:
              close = difflib.get_close_matches(stem + ".mp4", self.files, n=1, cutoff=0.5)
              if close:
                  video_path = os.path.join(self.video_dir, close[0])
              else:
                  print(f"Error: No video found for filename: {row['filename']}")
                  raise FileNotFoundError(f"No video for '{row['filename']}'") 
      try:
        frames = load_my_video(video_path)
        frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()
        enc = self.fe(images=list(frames_np), return_tensors="pt")

        return {
            "pixel_values": enc.pixel_values[0],
            "labels": torch.tensor(self.label2id[row["label"]], dtype=torch.long)
        }
      except Exception as e:
        print(f"Error processing video: {video_path}")
        print(f"Error message: {e}")
        raise e


def collate_video(batch):
    batch = [b for b in batch if b is not None]
    pixels = [b["pixel_values"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    return {"pixel_values": torch.stack(pixels), "labels": labels}



def freeze_last_two_and_head(model):
    for p in model.parameters(): p.requires_grad = False

    for p in model.classifier.parameters(): p.requires_grad = True

    modlists = [(name, m) for name, m in model.named_modules()
                if isinstance(m, torch.nn.ModuleList)]
    name, longest = max(modlists, key=lambda x: len(x[1]))
    print(f"Unfreezing last two layers from '{name}' (length={len(longest)})")

    for layer in list(longest)[-2:]:
        for p in layer.parameters(): p.requires_grad = True

    for p in model.classifier.parameters(): p.requires_grad = True
def train_model(model, train_loader, val_loader):
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"{model.name} Epoch {epoch} Train"):
            for k,v in batch.items(): batch[k] = v.to(DEVICE)
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += outputs.loss.item()
        model.eval()
        correct, total = 0, 0
        for batch in tqdm(val_loader, desc=f"{model.name} Epoch {epoch} Val"):
            for k,v in batch.items(): batch[k] = v.to(DEVICE)
            with torch.no_grad():
                logits = model(
                    **{k:batch[k] for k in batch if k!="labels"}
                ).logits
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            total   += batch["labels"].size(0)
        acc = correct/total
        tqdm.write(f"{model.name} Epoch {epoch} ▶ Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{model.name}_best.pth"))
    return model


@torch.no_grad()
def extract_and_save_embeddings(model, loader, split, df_subset):
    model.eval()
    model.config.output_hidden_states = True
    embs, labs, metadata = [], [], []

    for i, batch in enumerate(tqdm(loader, desc=f"{model.name} Embeds {split}")):
        for k, v in batch.items():
            batch[k] = v.to(DEVICE)

        out = model(**{k: batch[k] for k in batch if k != "labels"})
        hs = out.hidden_states[-1][:, 0, :].cpu()
        embs.append(hs)
        labs.append(batch["labels"].cpu())

        batch_indices = range(i * loader.batch_size, i * loader.batch_size + len(batch["labels"]))
        metadata.extend(df_subset.iloc[list(batch_indices)][['dialog_id', 'start_time', 'end_time', 'label']].to_dict(orient="records"))

    torch.save({
        "embeddings": torch.cat(embs, dim=0),
        "labels":     torch.cat(labs, dim=0),
        "metadata":   metadata
    }, os.path.join(OUTPUT_DIR, f"{model.name}_{split}_emb.pt"))

for seed in [42,43,44,45]:
    seed_everything(seed)
    for spec in MODEL_SPECS:
        print(f"\n▶▶▶ {spec['name']} ({spec['modality']})")

        if spec["modality"] == "text":
            processor = AutoTokenizer.from_pretrained(spec["checkpoint"])
            model = AutoModelForSequenceClassification.from_pretrained(
                spec["checkpoint"],
                num_labels=len(unique_labels),
                ignore_mismatched_sizes=True
            )
            DS, collate_fn = TextDataset, collate_text

        elif spec["modality"] == "audio":
            processor = AutoFeatureExtractor.from_pretrained(spec["checkpoint"])
            model = AutoModelForAudioClassification.from_pretrained(
                spec["checkpoint"],
                num_labels=len(unique_labels),
                ignore_mismatched_sizes=True
            )
            DS, collate_fn = AudioDataset, collate_audio
        else:  
            processor = AutoFeatureExtractor.from_pretrained(spec["checkpoint"])
            model = AutoModelForVideoClassification.from_pretrained(
                spec["checkpoint"],
                num_labels=len(unique_labels),
                ignore_mismatched_sizes=True
            )
            DS, collate_fn = VideoDataset, collate_video

        model.name = spec["name"]
        model.config.label2id = label2id
        model.config.id2label = id2label
        freeze_last_two_and_head(model)
        model.to(DEVICE)

        if spec["modality"] == "text":
            train_ds = DS(df_train, processor, label2id)
            val_ds   = DS(df_val,   processor, label2id)
            test_ds  = DS(df_test,  processor, label2id)
        elif spec["modality"] == "audio":
            train_ds = DS(df_train, AUDIO_DIR, processor, label2id)
            val_ds   = DS(df_val,   AUDIO_DIR, processor, label2id)
            test_ds  = DS(df_test,  AUDIO_DIR, processor, label2id)
        else:
            train_ds = DS(df_train, VIDEO_DIR, processor, label2id)
            val_ds   = DS(df_val,   VIDEO_DIR, processor, label2id)
            test_ds  = DS(df_test,  VIDEO_DIR, processor, label2id)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        trained = train_model(model, train_loader, val_loader)
        for split, loader, df_split in [("train", train_loader, df_train),("val",   val_loader,   df_val),("test",  test_loader,  df_test)]: extract_and_save_embeddings(trained, loader, split, df_split)
        drive_dir = f"/content/drive/MyDrive/multimodal_empathy/models/{seed}/{spec['name']}"
        os.makedirs(drive_dir, exist_ok=True)

        model_ckpt = os.path.join(OUTPUT_DIR, f"{spec['name']}_best.pth")
        train_emb  = os.path.join(OUTPUT_DIR, f"{spec['name']}_train_emb.pt")
        val_emb    = os.path.join(OUTPUT_DIR, f"{spec['name']}_val_emb.pt")
        test_emb   = os.path.join(OUTPUT_DIR, f"{spec['name']}_test_emb.pt")

OUTPUT_DIR = "/content/drive/MyDrive/multimodal_empathy/models/"
df_test = pd.read_csv("/content/drive/MyDrive/multimodal_empathy/dataset/test.csv")

print("Evaluating each model on test set (.pth):\n")
print(MODEL_SPECS)
for spec in MODEL_SPECS:
    name = spec["name"]
    ckpt = spec["checkpoint"]
    modality = spec["modality"]
    if modality == "text":
        processor = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt, num_labels=len(label2id), ignore_mismatched_sizes=True
        )
        test_ds = TextDataset(df_test, processor, label2id)
        collate_fn = collate_text

    elif modality == "audio":
        processor = AutoFeatureExtractor.from_pretrained(ckpt)
        model = AutoModelForAudioClassification.from_pretrained(
            ckpt, num_labels=len(label2id), ignore_mismatched_sizes=True
        )
        test_ds = AudioDataset(df_test, AUDIO_DIR, processor, label2id)
        collate_fn = collate_audio

    else:  
        processor = AutoFeatureExtractor.from_pretrained(ckpt)
        model = AutoModelForVideoClassification.from_pretrained(
            ckpt, num_labels=len(label2id), ignore_mismatched_sizes=True
        )
        test_ds = VideoDataset(df_test, VIDEO_DIR, processor, label2id)
        collate_fn = collate_video
    path = os.path.join(OUTPUT_DIR, f"{name}/{name}_best.pth")
    if not os.path.exists(path):
        print(f"{name} not there")
        continue

    model.load_state_dict(torch.load(path))
    model.to(DEVICE)
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    #eval 
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            for k,v in batch.items(): batch[k] = v.to(DEVICE)
            logits = model(**{k: batch[k] for k in batch if k != "labels"}).logits
            preds  = logits.argmax(-1)
            all_preds.append(preds.cpu())
            all_labels.append(batch["labels"].cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = accuracy_score(labels, preds)

    print(f"{name:15s} acc: {acc:.4f}")
"""Kode bersama untuk 04_pilot dan 05_training.

Isi: memuat data hasil tokenisasi, model late fusion, loop training, evaluasi,
pengukuran efisiensi, dan antrean run yang bisa dilanjutkan.
"""
import gc
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

MODELS = {"roberta": "roberta-base", "distilbert": "distilbert-base-cased"}
INPUT_COLS = {"comment": ["ids_comment"], "early": ["ids_pair"], "late": ["ids_parent", "ids_comment"]}

DEFAULTS = dict(
    epochs=3,
    batch_size=32,
    eval_batch_size=128,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=42,
    n_train_layers=None,     # None = semua layer dilatih (full fine-tuning); angka = hanya k layer teratas (P2)
    group_size=50,           # group_by_length: data diacak, lalu diurutkan per kelompok 50 batch
    select_metric="macro_f1",  # metrik validasi untuk memilih epoch terbaik
    log_points=10,           # berapa kali loss & learning rate dicatat ke train_log.csv per epoch
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------------------------------------------------------- data
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SarcasmData:
    """Token satu split untuk satu jenis input. Panjang dipakai untuk group_by_length."""

    def __init__(self, df, input_mode):
        self.mode = input_mode
        self.raw_idx = df["raw_idx"].to_numpy()
        self.labels = df["label"].to_numpy()
        self.cols = {c: df[c].to_numpy() for c in INPUT_COLS[input_mode]}
        self.lengths = sum(np.fromiter((len(x) for x in v), dtype=np.int64, count=len(v)) for v in self.cols.values())

    def __len__(self):
        return len(self.labels)


def load_split(data_dir, model_key, split, input_mode, limit=None, seed=42):
    cols = ["raw_idx", "label"] + INPUT_COLS[input_mode]
    df = pd.read_parquet(Path(data_dir) / model_key / f"{split}.parquet", columns=cols)
    if limit is not None and limit < len(df):
        df = df.groupby("label", group_keys=False).sample(limit // 2, random_state=seed)   # sampel seimbang
    return SarcasmData(df.reset_index(drop=True), input_mode)


def train_batches(lengths, batch_size, seed, group_size):
    """Acak data, urutkan per kelompok besar menurut panjang, lalu acak urutan batch."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(lengths))
    mega = batch_size * group_size
    batches = []
    for s in range(0, len(idx), mega):
        chunk = idx[s:s + mega]
        chunk = chunk[np.argsort(-lengths[chunk], kind="stable")]
        batches += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def eval_batches(lengths, batch_size):
    idx = np.argsort(-lengths, kind="stable")
    return [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]


def pad(seqs, pad_id):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.from_numpy(np.asarray(s, dtype=np.int64))
        mask[i, :len(s)] = 1
    return ids, mask


def make_batch(data, idx, pad_id, device):
    batch = {"labels": torch.as_tensor(data.labels[idx], dtype=torch.long)}
    if data.mode == "late":
        batch["parent_ids"], batch["parent_mask"] = pad(data.cols["ids_parent"][idx], pad_id)
        batch["comment_ids"], batch["comment_mask"] = pad(data.cols["ids_comment"][idx], pad_id)
    else:
        batch["input_ids"], batch["attention_mask"] = pad(data.cols[INPUT_COLS[data.mode][0]][idx], pad_id)
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# ----------------------------------------------------------------------------- model
class LateFusionClassifier(nn.Module):
    """Parent dan comment dibaca encoder yang sama, ringkasan = vektor token pertama,
    digabung [h_x; h_y; |h_x - h_y|; h_x * h_y], lalu diklasifikasi.
    Bentuk head mengikuti head bawaan masing-masing model, hanya ukuran input-nya 4x."""

    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        cfg = self.encoder.config
        h = cfg.hidden_size
        if cfg.model_type == "distilbert":      # pre_classifier -> ReLU -> dropout -> classifier
            self.head = nn.Sequential(nn.Linear(4 * h, h), nn.ReLU(), nn.Dropout(cfg.seq_classif_dropout), nn.Linear(h, 2))
        else:                                   # dropout -> dense -> tanh -> dropout -> out_proj
            drop = cfg.classifier_dropout if cfg.classifier_dropout is not None else cfg.hidden_dropout_prob
            self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(4 * h, h), nn.Tanh(), nn.Dropout(drop), nn.Linear(h, 2))
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=cfg.initializer_range)
                nn.init.zeros_(m.bias)

    @property
    def base_model(self):
        return self.encoder

    def summarize(self, ids, mask):
        return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]

    def forward(self, parent_ids, parent_mask, comment_ids, comment_mask):
        h_y = self.summarize(parent_ids, parent_mask)
        h_x = self.summarize(comment_ids, comment_mask)
        return self.head(torch.cat([h_x, h_y, (h_x - h_y).abs(), h_x * h_y], dim=-1))


def build_model(model_key, input_mode):
    if input_mode == "late":
        return LateFusionClassifier(MODELS[model_key])
    return AutoModelForSequenceClassification.from_pretrained(MODELS[model_key], num_labels=2)


def logits_of(model, batch, input_mode):
    if input_mode == "late":
        return model(batch["parent_ids"], batch["parent_mask"], batch["comment_ids"], batch["comment_mask"])
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits


def freeze_lower_layers(model, n_train_layers):
    """P2: bekukan encoder kecuali k layer teratas. Head klasifikasi selalu dilatih."""
    if n_train_layers is None:
        return
    base = model.base_model
    layers = base.encoder.layer if hasattr(base, "encoder") else base.transformer.layer
    for p in base.parameters():
        p.requires_grad = False
    if n_train_layers > 0:
        for layer in layers[-n_train_layers:]:
            for p in layer.parameters():
                p.requires_grad = True


def make_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 else decay).append(p)      # bias & LayerNorm tanpa weight decay
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def make_scheduler(optimizer, total_steps, warmup_steps):
    def factor(step):                       # naik linear selama warmup, lalu turun linear sampai 0
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ----------------------------------------------------------------------------- evaluasi
@torch.no_grad()
def predict(model, data, input_mode, pad_id, device, batch_size):
    model.eval()
    probs = np.zeros(len(data), dtype=np.float32)
    loss_sum = 0.0
    lossf = nn.CrossEntropyLoss(reduction="sum")
    for idx in eval_batches(data.lengths, batch_size):
        b = make_batch(data, idx, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = logits_of(model, b, input_mode)
        logits = logits.float()
        loss_sum += lossf(logits, b["labels"]).item()
        probs[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs, loss_sum / len(data)


def compute_metrics(labels, probs):
    pred = (probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", pos_label=1, zero_division=0)
    return {"accuracy": accuracy_score(labels, pred), "precision": p, "recall": r, "f1": f1,
            "macro_f1": f1_score(labels, pred, average="macro")}


@torch.no_grad()
def measure_latency(model, data, input_mode, pad_id, device, n=1000, seed=42):
    """Rata-rata waktu prediksi per data (ms), batch 1 dan batch 32, data diambil acak dari validasi."""
    model.eval()
    idx_all = np.random.default_rng(seed).choice(len(data), min(n, len(data)), replace=False)
    out = {}
    for bs in (1, 32):
        prepared = [make_batch(data, idx_all[i:i + bs], pad_id, device) for i in range(0, len(idx_all), bs)]
        for b in prepared[:10]:                                   # pemanasan GPU, tidak dihitung
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits_of(model, b, input_mode)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for b in prepared:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits_of(model, b, input_mode)
        torch.cuda.synchronize()
        out[bs] = (time.perf_counter() - t) / len(idx_all) * 1000
    return out


# ----------------------------------------------------------------------------- satu run
def _append_csv(row, path):
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_experiment(cfg, data_dir, out_dir, train_split="train", train_limit=None,
                   val_limit=None, test_limit=None, **overrides):
    """Latih satu konfigurasi. Run yang sudah ada di runs.csv dilewati.
    Jika berhenti di tengah jalan, run dilanjutkan dari epoch terakhir yang selesai."""
    cfg = {**DEFAULTS, **cfg, **overrides}
    rid = cfg["run_id"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv, hist_csv, log_csv = out_dir / "runs.csv", out_dir / "history.csv", out_dir / "train_log.csv"
    if runs_csv.exists() and rid in set(pd.read_csv(runs_csv)["run_id"]):
        print(f"[{rid}] sudah selesai, dilewati")
        return None

    device = "cuda"
    run_dir = out_dir / "checkpoints" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    pad_id = AutoTokenizer.from_pretrained(MODELS[cfg["model"]]).pad_token_id
    train = load_split(data_dir, cfg["model"], train_split, cfg["input"], train_limit, cfg["seed"])
    val = load_split(data_dir, cfg["model"], "val", cfg["input"], val_limit, cfg["seed"])

    set_seed(cfg["seed"])
    model = build_model(cfg["model"], cfg["input"])
    freeze_lower_layers(model, cfg["n_train_layers"])
    model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable = [p for p in model.parameters() if p.requires_grad]

    steps_per_epoch = math.ceil(len(train) / cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    optimizer = make_optimizer(model, cfg["lr"], cfg["weight_decay"])
    scheduler = make_scheduler(optimizer, total_steps, int(cfg["warmup_ratio"] * total_steps))

    state = {"epoch": 0, "best_score": -1.0, "best_epoch": None, "train_time": 0.0, "peak_mem_gb": 0.0}
    last_ckpt, best_ckpt = run_dir / "last.pt", run_dir / "best.pt"
    if last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        state = ck["state"]
        print(f"[{rid}] dilanjutkan dari epoch {state['epoch']}")
    for csv in (hist_csv, log_csv):                        # buang catatan dari epoch yang terputus
        if csv.exists():
            h = pd.read_csv(csv)
            h[~((h["run_id"] == rid) & (h["epoch"] > state["epoch"]))].to_csv(csv, index=False)

    print(f"[{rid}] train {len(train):,} | val {len(val):,} | {steps_per_epoch:,} langkah/epoch | "
          f"parameter {n_total / 1e6:.1f} juta (dilatih {n_trainable / 1e6:.1f} juta)")
    lossf = nn.CrossEntropyLoss()

    for epoch in range(state["epoch"] + 1, cfg["epochs"] + 1):
        model.train()
        batches = train_batches(train.lengths, cfg["batch_size"], cfg["seed"] + epoch, cfg["group_size"])
        n_log = cfg["log_points"]
        log_steps = {math.ceil(len(batches) * k / n_log) for k in range(1, n_log + 1)}
        loss_sum = torch.zeros((), device=device)
        win_loss, win_n = 0.0, 0                             # loss sejak titik catat sebelumnya
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for step, idx in enumerate(batches, 1):
            b = make_batch(train, idx, pad_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = logits_of(model, b, cfg["input"])
            loss = lossf(logits.float(), b["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.detach() * len(idx)
            if step in log_steps:
                el = time.perf_counter() - t0
                total = loss_sum.item()
                seen = sum(len(x) for x in batches[:step])
                win = (total - win_loss) / (seen - win_n)
                _append_csv({"run_id": rid, "epoch": epoch, "step": step, "global_step": (epoch - 1) * len(batches) + step,
                             "epoch_progress": epoch - 1 + step / len(batches), "train_loss": win,
                             "lr": scheduler.get_last_lr()[0]}, log_csv)
                win_loss, win_n = total, seen
                print(f"  epoch {epoch} | {step:>6,}/{len(batches):,} | loss {win:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                      f"{el / 60:.1f} mnt, sisa ±{el / step * (len(batches) - step) / 60:.1f} mnt", flush=True)
        torch.cuda.synchronize()
        t_train = time.perf_counter() - t0
        state["train_time"] += t_train
        state["peak_mem_gb"] = max(state["peak_mem_gb"], torch.cuda.max_memory_allocated() / 1e9)

        t1 = time.perf_counter()
        probs, val_loss = predict(model, val, cfg["input"], pad_id, device, cfg["eval_batch_size"])
        t_val = time.perf_counter() - t1
        m = compute_metrics(val.labels, probs)
        _append_csv({"run_id": rid, "epoch": epoch, "train_loss": loss_sum.item() / len(train), "val_loss": val_loss,
                     **m, "train_min": t_train / 60, "val_min": t_val / 60,
                     "train_samples_per_sec": len(train) / t_train, "val_samples_per_sec": len(val) / t_val}, hist_csv)
        print(f"  epoch {epoch} selesai | train {t_train / 60:.1f} mnt | val loss {val_loss:.4f} | "
              f"acc {m['accuracy']:.4f} P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f} macro-F1 {m['macro_f1']:.4f}")

        if m[cfg["select_metric"]] > state["best_score"]:
            state["best_score"], state["best_epoch"] = m[cfg["select_metric"]], epoch
            torch.save(model.state_dict(), best_ckpt)
        state["epoch"] = epoch
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "state": state}, last_ckpt)

    # model epoch terbaik -> prediksi test (disegel) + latensi
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    test = load_split(data_dir, cfg["model"], "test", cfg["input"], test_limit, cfg["seed"])
    t2 = time.perf_counter()
    test_probs, _ = predict(model, test, cfg["input"], pad_id, device, cfg["eval_batch_size"])
    t_test = time.perf_counter() - t2
    sealed = out_dir / "test_sealed"
    sealed.mkdir(exist_ok=True)
    pd.DataFrame({"raw_idx": test.raw_idx, "label": test.labels, "prob_sarkas": test_probs}).to_parquet(
        sealed / f"{rid}.parquet", index=False)
    lat = measure_latency(model, val, cfg["input"], pad_id, device, seed=cfg["seed"])

    h = pd.read_csv(hist_csv)
    best = h[(h["run_id"] == rid) & (h["epoch"] == state["best_epoch"])].iloc[0]
    row = {
        "run_id": rid, "round": cfg.get("round"), "input": cfg["input"], "model": cfg["model"], "lr": cfg["lr"],
        "seed": cfg["seed"], "n_train_layers": "all" if cfg["n_train_layers"] is None else cfg["n_train_layers"],
        "epochs": cfg["epochs"], "select_metric": cfg["select_metric"], "best_epoch": state["best_epoch"],
        **{f"val_{k}": best[k] for k in ["accuracy", "precision", "recall", "f1", "macro_f1"]},
        "val_loss": best["val_loss"],
        "params_total_m": n_total / 1e6, "params_trainable_m": n_trainable / 1e6,
        "train_min_total": state["train_time"] / 60, "train_min_per_epoch": state["train_time"] / 60 / cfg["epochs"],
        "train_samples_per_sec": len(train) * cfg["epochs"] / state["train_time"],
        "peak_mem_gb": state["peak_mem_gb"], "latency_bs1_ms": lat[1], "latency_bs32_ms": lat[32],
        "test_pred_min": t_test / 60, "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "batch_size": cfg["batch_size"], "precision_mode": "bf16", "gpu": torch.cuda.get_device_name(),
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    _append_csv(row, runs_csv)

    for f in (last_ckpt, best_ckpt):                       # checkpoint dihapus agar disk tidak penuh
        f.unlink(missing_ok=True)
    run_dir.rmdir()
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{rid}] SELESAI | epoch terbaik {state['best_epoch']} | val {cfg['select_metric']} {state['best_score']:.4f} | "
          f"training {state['train_time'] / 60:.1f} mnt | memori puncak {state['peak_mem_gb']:.2f} GB")
    return row


# ----------------------------------------------------------------------------- antrean
def build_queue(rounds, seed=42):
    """Putaran 0/1: 3 input x 2 model, LR 3e-5. Putaran 2: early fusion, LR lain.
    Putaran 3: comment & late fusion, LR lain."""
    queue = []

    def add(rnd, inp, model, lr):
        queue.append({"run_id": f"R{rnd}_{inp}_{model}_lr{lr:.0e}_s{seed}", "round": rnd,
                      "input": inp, "model": model, "lr": lr, "seed": seed})

    for rnd in rounds:
        if rnd in (0, 1):
            for inp in ["comment", "early", "late"]:
                for model in MODELS:
                    add(rnd, inp, model, 3e-5)
        elif rnd == 2:
            for lr in [1e-5, 2e-5, 5e-5]:
                for model in MODELS:
                    add(rnd, "early", model, lr)
        elif rnd == 3:
            for inp in ["comment", "late"]:
                for lr in [1e-5, 2e-5, 5e-5]:
                    for model in MODELS:
                        add(rnd, inp, model, lr)
    return queue


def run_status(queue, out_dir):
    runs_csv = Path(out_dir) / "runs.csv"
    done = set(pd.read_csv(runs_csv)["run_id"]) if runs_csv.exists() else set()
    return pd.DataFrame(queue).assign(status=lambda d: np.where(d["run_id"].isin(done), "selesai", "belum"))

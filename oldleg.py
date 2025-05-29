
#!/usr/bin/env python
# aecf_full_split.py  ·  standalone runner for COCO‑2014
# -----------------------------------------------------
# * deterministic split: 60 000 train / 5 000 val / 5 000 test
# * builds cached CLIP‑features for each bucket once
# * trains on train+val, early‑stops on val, evaluates on test
# * writes one JSON per run with **test‑set metrics**

import json, random, pathlib, shutil, time
from typing import Dict, List

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pycocotools.coco import COCO
import open_clip                               # CLIP backend
from PIL import Image
from tqdm.auto import tqdm
# where all JSONs will live
RESULTS_DIR = pathlib.Path("/content/coco_results_split")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)   # make sure it exists

# ------------------------------------------------------------------ #
# 0.  deterministic COCO split                                       #
# ------------------------------------------------------------------ #
ROOT = pathlib.Path("/content/coco2014")
SPLIT_JSON = ROOT / "splits_60k5k5k.json"
SEED = 42

def make_split():
    if SPLIT_JSON.exists():
        return json.loads(SPLIT_JSON.read_text())

    random.seed(SEED)
    def sample_ids(ann_file, k):
        coco = COCO(str(ann_file))
        ids  = list(coco.imgToAnns.keys())
        random.shuffle(ids)
        return ids[:k]

    train_ann = ROOT/"annotations"/"instances_train2014.json"
    val_ann   = ROOT/"annotations"/"instances_val2014.json"

    ids = sample_ids(train_ann, 65_000)      # 60k train + 5k val
    split = dict(
        train60k = ids[:60_000],
        val5k    = ids[60_000:],
        test5k   = sample_ids(val_ann, 5_000)
    )
    SPLIT_JSON.write_text(json.dumps(split, indent=2))
    print("✓ wrote", SPLIT_JSON)
    return split

# ------------------------------------------------------------------ #
# 1.  minimal Dataset & feature encoder                              #
# ------------------------------------------------------------------ #
COCO_THING_IDS = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,
                  22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,
                  41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,
                  62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,
                  84,85,86,87,88,89,90]
ID2IDX = {cid:i for i,cid in enumerate(COCO_THING_IDS)}
NUM_CLASSES = len(ID2IDX)

class CocoPairDataset(Dataset):
    """(img_path, caption, 80‑hot label) tuples restricted to a set of IDs."""
    def __init__(self, root, split_label, id_set):
        year = "2014"
        self.root = pathlib.Path(root)
        self.img_dir = self.root/f"{split_label}{year}"
        self.det = COCO(str(self.root/"annotations"/f"instances_{split_label}{year}.json"))
        self.cap = COCO(str(self.root/"annotations"/f"captions_{split_label}{year}.json"))
        self.ids = [i for i in self.det.imgToAnns.keys() if i in id_set]

    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        file_name = self.det.loadImgs(img_id)[0]["file_name"]
        caption   = self.cap.imgToAnns[img_id][0]["caption"]
        lab = torch.zeros(NUM_CLASSES, dtype=torch.float16)
        for ann in self.det.imgToAnns[img_id]:
            cid = ann["category_id"]
            if cid in ID2IDX: lab[ID2IDX[cid]] = 1.
        return str(self.img_dir/file_name), caption, lab

def build_cache(root, subset, ds, clip_arch="ViT-B-32", clip_pretrained="openai",
                batch_gpu=512, num_workers=8):
    dest = root/f"{subset}_clip_feats.pt"
    if dest.exists():
        print("⏩", dest.name, "already exists")
        return dest
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_arch, pretrained=clip_pretrained)
    model = model.cuda().eval().requires_grad_(False)
    tok   = open_clip.get_tokenizer(clip_arch)

    dl = DataLoader(ds, batch_size=batch_gpu, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                    collate_fn=lambda b: list(zip(*b)))
    feats_i, feats_t, labs = [], [], []
    for img_paths, caps, lab_batch in tqdm(dl, leave=False, desc=dest.name):
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB"))
                            for p in img_paths]).cuda(non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            fi = model.encode_image(imgs)
            ft = model.encode_text(tok(caps).cuda(non_blocking=True))
        feats_i.append(fi.cpu()); feats_t.append(ft.cpu()); labs.append(torch.stack(lab_batch))
    obj = dict(img=torch.cat(feats_i).bfloat16(),
               txt=torch.cat(feats_t).bfloat16(),
               y  =torch.cat(labs))
    torch.save(obj, dest)
    print("✓ saved", dest, f"({len(ds):,})")
    return dest

# ------------------------------------------------------------------ #
# 2.  tiny Lightning model (gate + CLS)                              #
# ------------------------------------------------------------------ #
class GatingNet(nn.Module):
    def __init__(self, dim):
        super().__init__();
        self.fc=nn.Sequential(nn.Linear(dim*2,256),nn.ReLU(),nn.Linear(256,2))
    def forward(self,f): return F.softmax(self.fc(f),-1)

class AECF(pl.LightningModule):
    def __init__(self, num_classes, entropy_max=0.5, cec_coef=0.5, entropy_free=2, entropy_warmup=5):
        super().__init__()
        self.save_hyperparameters()
        self.gate = GatingNet(512)
        self.cls  = nn.Linear(512, num_classes)
    def forward(self, fi, ft):
        w = self.gate(torch.cat([fi,ft],-1))
        f = w[:,0:1]*fi + w[:,1:2]*ft
        return self.cls(f)
    def training_step(self,b,idx):
        fi,ft,y = b["image"],b["text"],b["label"]
        logits = self(fi,ft)
        loss   = F.binary_cross_entropy_with_logits(logits,y.float())
        self.log("train_loss",loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
    def validation_step(self, batch, idx):
        fi, ft, y = batch["image"], batch["text"], batch["label"]
        logits = self(fi, ft)
        vloss  = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log("val_loss", vloss, prog_bar=True)   # <-- this is what the callback monitors
# ------------------------------------------------------------------ #
# 3.  DataModule for cached tensors                                  #
# ------------------------------------------------------------------ #
class ClipTensor(Dataset):
    def __init__(self,obj): self.img=obj["img"]; self.txt=obj["txt"]; self.y=obj["y"]
    def __len__(self): return self.y.size(0)
    def __getitem__(self,i): return {"image":self.img[i],"text":self.txt[i],"label":self.y[i]}

class DM(pl.LightningDataModule):
    def __init__(self, path, batch=512):
        super().__init__(); self.path=pathlib.Path(path); self.batch=batch
    def setup(self,stage=None):
        tr=torch.load(self.path/"train_60k_clip_feats.pt")
        va=torch.load(self.path/"val_5k_clip_feats.pt")
        self.train = ClipTensor(tr); self.val = ClipTensor(va)
    def _dl(self,ds,sh): return DataLoader(ds,self.batch,shuffle=sh,num_workers=4,pin_memory=True)
    def train_dataloader(self): return self._dl(self.train,True)
    def val_dataloader(self):   return self._dl(self.val,False)

# ------------------------------------------------------------------ #
# 4.  Training + test evaluation                                     #
# ------------------------------------------------------------------ #
def run_once(run_label:str):
    split = make_split()
    # build caches
    
    build_cache(ROOT,"train_60k", CocoPairDataset(ROOT,"train",split["train60k"]))
    build_cache(ROOT,"val_5k",   CocoPairDataset(ROOT,"train",split["val5k"]))
    build_cache(ROOT,"test_5k",  CocoPairDataset(ROOT,"val",  split["test5k"]))

    dm = DM(ROOT, batch=512)
    model = AECF(num_classes=80)
    ckpt = ModelCheckpoint(dirpath="/tmp",monitor="val_loss",mode="min",save_top_k=1)
    Trainer(accelerator="gpu",devices=1,max_epochs=15,precision="bf16-mixed",
            callbacks=[ckpt]).fit(model,dm)

    best = AECF.load_from_checkpoint(ckpt.best_model_path).cuda().eval()

    test_ds = ClipTensor(torch.load(ROOT/"test_5k_clip_feats.pt"))
    test_dl = DataLoader(test_ds,512,shuffle=False,num_workers=4)

    hits=tot=0
    from torch.cuda.amp import autocast

    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        for b in test_dl:
            img = b["image"].cuda()
            txt = b["text"].cuda()
            p = torch.sigmoid(best(img, txt)[0])    # ✔ dtypes now match
            y=b["label"].cuda()
            hits+=((p>0.5)&y.bool()).any(1).sum().item(); tot+=y.size(0)
    rec = hits/tot
    out_path = RESULTS_DIR / f"{run_label}.json"
    out_path.write_text(json.dumps(dict(rec1_test=rec), indent=2))
    print("Test Rec@1", rec, "→", out_path)


if __name__ == "__main__":
    run_once("quick_demo")

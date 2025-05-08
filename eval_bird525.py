#!/usr/bin/env python3
"""
eval_birdsnap.py

Load local Birdsnap .arrow shards via Arrow, extract features with your frozen CvT or ViT encoder,
report 1-NN generalization accuracy plus precision, recall, F1, and a classification report.
Supports CvT variants and ViT variants (base/large).
"""
import os
import io
import glob
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from datasets import load_dataset
import timm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import time
from tqdm import tqdm
from PIL import Image, ImageFile
from einops import rearrange

ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils.custom_cvt import ConvolutionalVisionTransformer as CustomCvT

class CvTEncoder(nn.Module):
    def __init__(self, cvt_model):
        super().__init__()
        self.cvt_model = cvt_model
    def forward(self, x):
        for i in range(self.cvt_model.num_stages):
            x, cls_tokens, _ = getattr(self.cvt_model, f"stage{i}")(x)
        if self.cvt_model.cls_token:
            return torch.squeeze(self.cvt_model.norm(cls_tokens), dim=1)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.cvt_model.norm(x)
        return x.mean(dim=1)

class FeatureCvT(nn.Module):
    def __init__(self, cvt: CustomCvT):
        super().__init__()
        self.encoder = CvTEncoder(cvt)
        # detect embed dim
        if hasattr(cvt, 'spec') and 'DIM_EMBED' in cvt.spec:
            embed_dim = cvt.spec['DIM_EMBED'][-1]
        else:
            try:
                embed_dim = cvt.stage2.norm2.normalized_shape[0]
            except:
                embed_dim = 1024
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        f = self.encoder(x)
        return self.norm(f)

class FeatureViT(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        feat_dim = getattr(vit_model, 'num_features', vit_model.embed_dim)
        self.norm = nn.LayerNorm(feat_dim, eps=1e-6)
    def forward(self, x):
        f = self.vit.forward_features(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        if f.ndim == 3:
            f = f.mean(dim=1)
        return self.norm(f)

# ---- Data loader via Arrow builder ----
def make_loader(data_dir, batch_size):
    shards = sorted(glob.glob(os.path.join(data_dir, '*.arrow')))
    if not shards:
        raise FileNotFoundError(f"No .arrow shards found in {data_dir}")
    ds = load_dataset('arrow', data_files={'test': shards})['test']
    tf = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    class BirdsnapDataset(Dataset):
        def __init__(self, ds, tf):
            self.ds = ds
            self.tf = tf
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            ex = self.ds[idx]
            img, lbl = ex['image'], ex['label']
            if isinstance(img, list): img = img[0]
            if isinstance(img, (bytes, bytearray)):
                img = Image.open(io.BytesIO(img)).convert('RGB')
            elif isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, Image.Image):
                img = img.convert('RGB')
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
            return self.tf(img), lbl
    dataset = BirdsnapDataset(ds, tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    labels = np.array(ds['label'])
    return loader, labels

# ---- Model build ----
def build_cvt(weights_path, device, variant='auto', stride_kv=1):
    ckpt = torch.load(weights_path, map_location='cpu')

    if variant == 'auto':
        ed = next((v.shape[0] for k,v in ckpt.items() if 'attn.proj.weight' in k and 'stage2' in k), None)
        variant = 'small' if ed==384 else 'large' if ed==1024 else 'xsmall'
        print(f"Detected embedding dim {ed}, using variant '{variant}'")

    cfg = {
        'xsmall': {'DIM_EMBED':[64,192,384], 'DEPTH':[1,2,10], 'NUM_HEADS':[1,3,6], 'DROP_PATH_RATE':[0.0,0.0,0.1]},
        'small':  {'DIM_EMBED':[64,192,384], 'DEPTH':[1,4,16], 'NUM_HEADS':[1,3,6], 'DROP_PATH_RATE':[0.0,0.0,0.1]},
        'large':  {'DIM_EMBED':[192,768,1024], 'DEPTH':[2,2,20], 'NUM_HEADS':[3,12,16], 'DROP_PATH_RATE':[0.0,0.0,0.3]},
    }[variant]
    spec = dict(
        NUM_STAGES=3,
        PATCH_SIZE=[7,3,3], PATCH_STRIDE=[4,2,2], PATCH_PADDING=[2,1,1],
        DIM_EMBED=cfg['DIM_EMBED'], DEPTH=cfg['DEPTH'], NUM_HEADS=cfg['NUM_HEADS'],
        MLP_RATIO=[4.0]*3, QKV_BIAS=[True]*3, DROP_RATE=[0.0]*3, ATTN_DROP_RATE=[0.0]*3,
        DROP_PATH_RATE=cfg['DROP_PATH_RATE'], CLS_TOKEN=[False,False,True],
        QKV_PROJ_METHOD=['dw_bn']*3, KERNEL_QKV=[3]*3,
        PADDING_KV=[1]*3, STRIDE_KV=[stride_kv]*3,
        PADDING_Q=[1]*3, STRIDE_Q=[1]*3
    )
    cvt = CustomCvT(spec=spec); cvt.head = nn.Identity()
    model = FeatureCvT(cvt).to(device)
    st = model.state_dict()
    for k,v in ckpt.items():
        if k.startswith('image_encoder.cvt_model.'):
            nk = k.replace('image_encoder.cvt_model.', 'encoder.cvt_model.')
            if nk in st: st[nk] = v
    model.load_state_dict(st)
    ed_act = model.encoder.cvt_model.norm.weight.shape[0]
    if model.norm.normalized_shape[0] != ed_act:
        model.norm = nn.LayerNorm(ed_act, eps=1e-6).to(device)
    model.eval()
    return model


def _interpolate_pos_embed(state_pe, model_pe):
    if state_pe.shape == model_pe.shape:
        return state_pe
    cls, grid = state_pe[:,:1], state_pe[:,1:]
    gs_old = int(grid.size(1)**0.5); gs_new = int((model_pe.size(1)-1)**0.5)
    grid = grid.reshape(1, gs_old, gs_old, -1).permute(0,3,1,2)
    grid = nn.functional.interpolate(grid, size=(gs_new,gs_new), mode='bilinear', align_corners=False)
    grid = grid.permute(0,2,3,1).reshape(1, gs_new*gs_new, -1)
    return torch.cat((cls, grid), dim=1)

def build_vit(weights_path, device, variant='base'):
    name = 'vit_base_patch16_224' if variant=='base' else 'vit_large_patch16_224'
    vit = timm.create_model(name, img_size=448, pretrained=False); vit.head=nn.Identity()
    state = torch.load(weights_path, map_location='cpu')
    if any(k.startswith('image_encoder.') for k in state):
        state = {k.replace('image_encoder.', ''):v for k,v in state.items() if k.startswith('image_encoder.')}
    # interpolate pos_embed
    if 'pos_embed' in state and vit.pos_embed is not None:
        pe_ckpt, pe_mod = state['pos_embed'], vit.pos_embed
        if pe_ckpt.shape != pe_mod.shape:
            state['pos_embed'] = _interpolate_pos_embed(pe_ckpt, pe_mod)
            print(f"🔄 Interpolated pos_embed {tuple(pe_ckpt.shape)}→{tuple(pe_mod.shape)}")
    miss, unexp = vit.load_state_dict(state, strict=False)
    if miss: print(f"✅ Missing ViT keys: {len(miss)}")
    if unexp: print(f"✅ Stripped ViT keys: {len(unexp)}")
    model = FeatureViT(vit).to(device); model.eval()
    return model

# ---- Extract embeddings ----
def extract_feats(model, loader, device):
    feats=[]; pbar=tqdm(total=len(loader.dataset), desc='Extracting', unit='img')
    with torch.no_grad():
        for imgs,_ in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            feats.append(out.cpu().numpy()); pbar.update(imgs.size(0))
    pbar.close(); return np.vstack(feats)


def nearest_centroid(feats, labels, metric='cosine'):
    classes = np.unique(labels)

    # 1) compute mean (centroid) embedding for each class
    centroids = np.vstack([
        feats[labels == c].mean(axis=0)
        for c in classes
    ])

    # 2) fit a 1-NN on those centroids
    nbrs = NearestNeighbors(n_neighbors=1, metric=metric).fit(centroids)

    # 3) find for all samples the nearest centroid
    dists, idxs = nbrs.kneighbors(feats)

    # 4) map back to class labels
    preds = classes[idxs[:,0]]
    return preds, labels


# ---- Main ----
def main():
    parser=argparse.ArgumentParser(description='1-NN eval on Birdsnap')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--model_type', choices=['cvt','vit'], default='cvt')
    parser.add_argument('--cvt_variant', choices=['xsmall','small','large','auto'], default='auto')
    parser.add_argument('--vit_variant', choices=['base','large'], default='base')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--output_json', type=str, default=None)
    args = parser.parse_args()

    print(f"Loading shards from {args.data_dir}…")
    loader, labels = make_loader(args.data_dir, args.batch_size)
    print(f"Loaded {len(labels)} samples")

    if args.max_samples:
        N=min(args.max_samples, len(loader.dataset))
        loader = DataLoader(Subset(loader.dataset, list(range(N))), batch_size=args.batch_size, shuffle=False, num_workers=4)
        labels = labels[:N]; print(f"⚡ Subsampled to first {N} samples")

    print(f"Building {args.model_type.upper()} model…")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_type=='cvt':
        model = build_cvt(args.weights, device, args.cvt_variant)
    else:
        model = build_vit(args.weights, device, args.vit_variant)
    print(f"Model ready on {device}")

    print("Extracting features…")
    start=time.time()
    feats = extract_feats(model, loader, device)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    elapsed = time.time() - start
    throughput = len(loader.dataset) / elapsed
    print(f"Throughput: {throughput:.2f} img/s over {len(loader.dataset)} imgs in {elapsed:.2f}s")

    print("Running 1-NN evaluation…")
    preds, trues = nearest_centroid(feats, labels)

    acc  = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, average='macro', zero_division=0)
    rec  = recall_score(trues, preds, average='macro', zero_division=0)
    f1v  = f1_score(trues, preds, average='macro', zero_division=0)
    rep  = classification_report(trues, preds, zero_division=0)

    print(f"\nResults on Birdsnap ({len(labels)} samples):")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1v:.4f}\n")
    print(rep)

    if args.output_json:
        out = {
            'num_samples': len(labels),
            'accuracy':     acc,
            'precision':    prec,
            'recall':       rec,
            'f1_score':     f1v,
            'throughput':   throughput,
            'elapsed':      elapsed,
            'report':       rep
        }
        with open(args.output_json,'w') as f:
            json.dump(out, f, indent=4)
        print(f"Results + throughput saved to {args.output_json}")

if __name__=='__main__':
    main()

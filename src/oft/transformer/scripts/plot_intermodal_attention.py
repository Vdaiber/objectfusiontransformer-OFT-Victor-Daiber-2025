#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn as nn
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import logging
from omegaconf import OmegaConf
from oft.transformer.models.fusion_transformer_staged import ObjectFusionTransformerModelStaged
from oft.transformer.datasets.loader_staged import build_dataloaders_staged


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Inter-Modal Attention Heatmap")
    parser.add_argument("--config", type=str, required=True, help="Path to pipeline_staged.yaml")
    parser.add_argument("--batch_idx", type=int, default=0, help="Batch index to inspect")
    parser.add_argument("--time_idx", type=int, default=0, help="Time step to inspect (0..T-1)")
    parser.add_argument("--output", type=str, default="intermodal_attention.png", help="Output image path")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Config & DataLoader ---
    cfg = OmegaConf.load(args.config)
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    logger = logging.getLogger("plot_attn")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)

    dataloaders = build_dataloaders_staged(cfg_dict, logger)
    batch = next(iter(dataloaders['train']))

    # --- Modell init & Einzelauswahl ---
    model = ObjectFusionTransformerModelStaged(cfg_dict)
    model.eval()

    sensor_names = [
        s['name'] for s in cfg_dict['dataset']['virtual_sensors']
        if s.get('enabled', True)
    ]
    b = args.batch_idx
    t = args.time_idx

    # Extrahiere nur den gewählten Time-Step
    single = {}
    for name in sensor_names:
        sd = batch['sensor_data'][name]
        # jeweils (1, N, feat_dim) statt (B, T, N, feat_dim)
        single[name] = {
            'features'  : sd['features'][b:b+1, t],    # (1,N,10)
            'metadata': sd['metadata'][b:b+1, t],  # (1,N,2)
            'centers'      : sd['centers'][b:b+1, t],        # (1,N,3)
            'mask'     : sd['mask'][b:b+1, t]        # (1,N)
        }

    # --- Intra-Modal Encoding + Metadaten-Cross-Attention ---
    all_feats_list, all_pads = [], []
    for name in sensor_names:
        sd = single[name]
        obj = sd['features']      # (1,N,10)
        meta = sd['metadata']   # (1,N,2)
        xyz = sd['centers']          # (1,N,3)
        pad = sd['mask']         # (1,N)

        # Projektion + PE-Encoder
        proj = model.input_proj[name](obj)  # (1,N,D)
        q = model.intra_modal_encoders[name](
            sensor_features=proj, 
            xyz_centers=xyz, 
            sensor_padding_mask=pad
        )                                   # (1,N,D)

        # Metadata-Cross-Attn
        enc_meta = model.metadata_encoder(meta)  # (1,N,D)
        refined, _ = model.metadata_cross_attention[name](
            query=q, key=enc_meta, value=enc_meta,
            key_padding_mask=pad
        )                                       # (1,N,D)

        all_feats_list.append(refined[0])  # [(N,D), ...]
        all_pads.append(pad[0])           # [(N,), ...]

    all_feats = torch.cat(all_feats_list, dim=0)    # (N_total, D)
    all_pad   = torch.cat(all_pads, dim=0)          # (N_total,)

    # --- Query-Feature-Cross-Attention extrahieren ---
    # Fusion-Queries: (Q, D)
    queries = model.inter_modal_fusion.fusion_queries.weight
    Q = queries.shape[0]

    # MultiheadAttention-Instanz
    attn_mha = model.inter_modal_fusion.query_aggregator.multihead_attn

    # Input für MHA muss (L, B, E)
    q = queries.unsqueeze(1)       # (Q, 1, D)
    k = all_feats.unsqueeze(1)     # (N_total, 1, D)
    v = all_feats.unsqueeze(1)

    # Explizit detach → kein Autograd-Problem beim numpy()
    with torch.no_grad():
        attn_out, attn_w = attn_mha(
            query=q, key=k, value=v,
            key_padding_mask=all_pad.unsqueeze(0),
            need_weights=True,
            average_attn_weights=False
        )
        # attn_w: Tensor(B=1, num_heads, Q, N_total)
        aw = attn_w.detach()[0].cpu().numpy()  # (num_heads, Q, N_total)

    # --- Aggregiere pro Sensor ---
    boundaries = np.cumsum([f.shape[0] for f in all_feats_list])
    boundaries = [0] + boundaries.tolist()
    sensor_attn = np.zeros((Q, len(sensor_names)))
    # mittlere Attention über alle Heads
    for i in range(len(sensor_names)):
        start, end = boundaries[i], boundaries[i+1]
        sensor_attn[:, i] = aw[:, :, start:end].mean(axis=(0, 2))

    # --- Plot ---
    plt.figure(figsize=(6, 10))
    plt.imshow(sensor_attn, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Avg Attention')
    plt.yticks(np.arange(Q))
    plt.xticks(np.arange(len(sensor_names)), sensor_names, rotation=45)
    plt.xlabel('Sensor Stream')
    plt.ylabel('Fusion Query Index')
    plt.title(f'Inter-Modal Attention (batch={b}, time={t})')
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output)
    print(f"✅ Saved attention heatmap to {args.output}")


if __name__ == '__main__':
    main()
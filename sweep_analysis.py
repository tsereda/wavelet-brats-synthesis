#!/usr/bin/env python3
"""
Sweep Ablation Analysis Script
Analyzes Architecture × Method × Wavelet ablation study results from W&B export.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
INPUT_CSV = "wandb_export_2026-01-13T09_18_27.907-06_00.csv"
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

plt.style.use('seaborn-v0_8-whitegrid')


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['config'] = df['architecture'] + '_' + df['model_mode'] + '_' + df['wavelet']
    for col in ['val/mse', 'val/psnr', 'val/ssim', 'train/step_time']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def create_combined_figure(df: pd.DataFrame):
    # Exclude swin + diffusion_fast combo
    exclude_mask = (df['architecture'] == 'swin') & (df['model_mode'] == 'diffusion_fast')
    df_filt = df[~exclude_mask].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    metrics = [
        ('val/mse', 'MSE (↓)', 'Reds', '.4f'),
        ('val/psnr', 'PSNR dB (↑)', 'Greens', '.1f'),
        ('val/ssim', 'SSIM (↑)', 'Greens', '.3f'),
    ]
    
    # Top row: by config (arch_method x wavelet)
    for ax, (metric, title, cmap, fmt) in zip(axes[0], metrics):
        agg = df_filt.groupby(['architecture', 'model_mode', 'wavelet'])[metric].mean().reset_index()
        agg['arch_method'] = agg['architecture'] + '_' + agg['model_mode']
        pivot = agg.pivot(index='arch_method', columns='wavelet', values=metric)
        pivot = pivot[[c for c in ['nowavelet', 'haar', 'db2'] if c in pivot.columns]]
        
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax, linewidths=0.5,
                    cbar_kws={'shrink': 0.8})
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Wavelet')
        ax.set_ylabel('')
    
    # Bottom row: by modality (modality x wavelet), averaged across arch/method
    for ax, (metric, title, cmap, fmt) in zip(axes[1], metrics):
        agg = df_filt.groupby(['contr', 'wavelet'])[metric].mean().reset_index()
        pivot = agg.pivot(index='contr', columns='wavelet', values=metric)
        pivot = pivot[[c for c in ['nowavelet', 'haar', 'db2'] if c in pivot.columns]]
        # Reorder modalities
        modality_order = ['t1n', 't1c', 't2w', 't2f']
        pivot = pivot.reindex([m for m in modality_order if m in pivot.index])
        
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax, linewidths=0.5,
                    cbar_kws={'shrink': 0.8})
        ax.set_title(f'{title} by Modality', fontweight='bold', fontsize=12)
        ax.set_xlabel('Wavelet')
        ax.set_ylabel('')
    
    plt.suptitle('Fast-cWDM Ablation Study', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    print("Loading data...")
    df = load_and_clean_data(INPUT_CSV)
    print(f"Loaded {len(df)} runs ({df['State'].value_counts().to_dict()})")
    
    fig = create_combined_figure(df)
    fig.savefig(OUTPUT_DIR / "sweep_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/sweep_analysis.png")
    
    df_fin = df[df['State'] == 'finished']
    if len(df_fin) > 0:
        best = df_fin.loc[df_fin['val/psnr'].idxmax()]
        print(f"\nBest: {best['val/psnr']:.2f} dB — {best['config']} ({best['contr']})")


if __name__ == "__main__":
    main()
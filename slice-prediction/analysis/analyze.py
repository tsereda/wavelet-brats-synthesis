"""
Wavelet Performance Analysis Script
Analyzes and visualizes wavelet transform performance on medical imaging data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

# Configuration
DEFAULT_INPUT_FILE = 'wandb_export_2025-10-26T19_05_40_002-05_00.csv'
OUTPUT_VIZ = 'wavelet_analysis.png'

def get_input_file():
    """Get input file from command line args or search for CSV files"""
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if os.path.exists(input_file):
            return input_file
        else:
            print(f"❌ Error: File '{input_file}' not found!")
            sys.exit(1)
    
    # Search for CSV files in current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ Error: No CSV files found in current directory!")
        print("\n📋 Usage:")
        print("   python analyze.py <path_to_csv_file>")
        print("\n   Example:")
        print("   python analyze.py wandb_export_2025-10-26T19_05_40.002-05_00.csv")
        sys.exit(1)
    
    if len(csv_files) == 1:
        print(f"✅ Found CSV file: {csv_files[0]}")
        return csv_files[0]
    else:
        print(f"❌ Multiple CSV files found. Please specify which one to use:")
        for i, f in enumerate(csv_files, 1):
            print(f"   {i}. {f}")
        print("\n📋 Usage: python analyze.py <filename>")
        sys.exit(1)

def load_and_clean_data(filepath):
    """Load CSV and remove incomplete experiments"""
    print(f"📂 Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    metric_cols = [col for col in df.columns if col != 'Name']
    df_clean = df.dropna(subset=metric_cols, how='all').copy()
    df_clean['wavelet_type'] = df_clean['Name'].str.extract(r'wavelet_([a-z0-9.]+)_\d+')[0]
    
    print(f"Total experiments: {len(df)}")
    print(f"Complete experiments: {len(df_clean)}")
    print(f"\nWavelet types found: {sorted(df_clean['wavelet_type'].unique())}")
    
    return df_clean

def create_visualizations(df_clean, output_file):
    """Generate comprehensive visualization plots"""
    modalities = ['flair', 't1', 't1ce', 't2']
    
    # Set up the plotting style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Wavelet Performance Analysis - Medical Image Reconstruction', 
                 fontsize=16, fontweight='bold')

    # 1. SSIM Comparison by Modality
    ax1 = axes[0, 0]
    ssim_cols = [f'eval/ssim_{mod}_mean' for mod in modalities]
    df_ssim = df_clean[['wavelet_type'] + ssim_cols].set_index('wavelet_type')
    df_ssim.columns = [col.replace('eval/ssim_', '').replace('_mean', '').upper() 
                       for col in df_ssim.columns]
    df_ssim.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('SSIM by Wavelet Type and Modality (Higher is Better)', fontweight='bold')
    ax1.set_xlabel('Wavelet Type')
    ax1.set_ylabel('SSIM Score')
    ax1.legend(title='Modality', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # 2. MSE Comparison by Modality
    ax2 = axes[0, 1]
    mse_cols = [f'eval/mse_{mod}_mean' for mod in modalities]
    df_mse = df_clean[['wavelet_type'] + mse_cols].set_index('wavelet_type')
    df_mse.columns = [col.replace('eval/mse_', '').replace('_mean', '').upper() 
                      for col in df_mse.columns]
    df_mse.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('MSE by Wavelet Type and Modality (Lower is Better)', fontweight='bold')
    ax2.set_xlabel('Wavelet Type')
    ax2.set_ylabel('MSE')
    ax2.legend(title='Modality', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    # 3. Average SSIM across all modalities
    ax3 = axes[1, 0]
    df_clean['avg_ssim'] = df_clean[ssim_cols].mean(axis=1)
    df_avg_ssim = df_clean[['wavelet_type', 'avg_ssim']].sort_values('avg_ssim', ascending=False)
    bars = ax3.barh(df_avg_ssim['wavelet_type'], df_avg_ssim['avg_ssim'], 
                    color=sns.color_palette("RdYlGn", len(df_avg_ssim)))
    ax3.set_title('Average SSIM Across All Modalities (Ranked)', fontweight='bold')
    ax3.set_xlabel('Average SSIM')
    ax3.set_ylabel('Wavelet Type')
    ax3.grid(axis='x', alpha=0.3)
    for i, (idx, row) in enumerate(df_avg_ssim.iterrows()):
        ax3.text(row['avg_ssim'] + 0.002, i, f"{row['avg_ssim']:.4f}", 
                va='center', fontsize=9)

    # 4. Average MSE across all modalities
    ax4 = axes[1, 1]
    df_clean['avg_mse'] = df_clean[mse_cols].mean(axis=1)
    df_avg_mse = df_clean[['wavelet_type', 'avg_mse']].sort_values('avg_mse', ascending=True)
    bars = ax4.barh(df_avg_mse['wavelet_type'], df_avg_mse['avg_mse'], 
                    color=sns.color_palette("RdYlGn_r", len(df_avg_mse)))
    ax4.set_title('Average MSE Across All Modalities (Ranked)', fontweight='bold')
    ax4.set_xlabel('Average MSE')
    ax4.set_ylabel('Wavelet Type')
    ax4.grid(axis='x', alpha=0.3)
    for i, (idx, row) in enumerate(df_avg_mse.iterrows()):
        ax4.text(row['avg_mse'] + 0.00002, i, f"{row['avg_mse']:.6f}", 
                va='center', fontsize=9)

    # 5. SSIM vs MSE Scatter (Trade-off analysis)
    ax5 = axes[2, 0]
    scatter = ax5.scatter(df_clean['avg_mse'], df_clean['avg_ssim'], 
                          s=200, alpha=0.7, c=range(len(df_clean)), 
                          cmap='viridis', edgecolors='black', linewidth=1.5)
    for idx, row in df_clean.iterrows():
        ax5.annotate(row['wavelet_type'], 
                    (row['avg_mse'], row['avg_ssim']),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=9, fontweight='bold')
    ax5.set_title('SSIM vs MSE Trade-off (Ideal: Top-Left)', fontweight='bold')
    ax5.set_xlabel('Average MSE (Lower is Better)')
    ax5.set_ylabel('Average SSIM (Higher is Better)')
    ax5.grid(True, alpha=0.3)

    # 6. Heatmap of all metrics
    ax6 = axes[2, 1]
    heatmap_data = df_clean[['wavelet_type'] + ssim_cols + mse_cols].set_index('wavelet_type')
    
    # Normalize metrics (SSIM: higher is better, MSE: lower is better - invert)
    heatmap_normalized = heatmap_data.copy()
    for col in ssim_cols:
        heatmap_normalized[col] = ((heatmap_data[col] - heatmap_data[col].min()) / 
                                   (heatmap_data[col].max() - heatmap_data[col].min()))
    for col in mse_cols:
        heatmap_normalized[col] = (1 - (heatmap_data[col] - heatmap_data[col].min()) / 
                                   (heatmap_data[col].max() - heatmap_data[col].min()))

    heatmap_normalized.columns = ['SSIM_FLAIR', 'SSIM_T1', 'SSIM_T1CE', 'SSIM_T2', 
                                   'MSE_FLAIR', 'MSE_T1', 'MSE_T1CE', 'MSE_T2']
    sns.heatmap(heatmap_normalized, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax6, 
                cbar_kws={'label': 'Normalized Score (1=Best)'}, vmin=0, vmax=1)
    ax6.set_title('Normalized Performance Heatmap (1=Best, 0=Worst)', fontweight='bold')
    ax6.set_xlabel('Metric')
    ax6.set_ylabel('Wavelet Type')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_file}")
    
    return df_clean

def print_analysis(df_clean):
    """Print detailed performance analysis"""
    modalities = ['flair', 't1', 't1ce', 't2']
    
    # Calculate rankings
    df_ranked = df_clean[['wavelet_type', 'avg_ssim', 'avg_mse']].copy()
    df_ranked['ssim_rank'] = df_ranked['avg_ssim'].rank(ascending=False)
    df_ranked['mse_rank'] = df_ranked['avg_mse'].rank(ascending=True)
    df_ranked['combined_rank'] = (df_ranked['ssim_rank'] + df_ranked['mse_rank']) / 2
    df_ranked = df_ranked.sort_values('combined_rank')

    print("\n" + "=" * 80)
    print("WAVELET PERFORMANCE RANKING (Combined SSIM + MSE)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Wavelet':<12} {'Avg SSIM':<12} {'Avg MSE':<14} {'Combined Score':<15}")
    print("-" * 80)
    for i, (idx, row) in enumerate(df_ranked.iterrows(), 1):
        print(f"{i:<6} {row['wavelet_type']:<12} {row['avg_ssim']:<12.6f} "
              f"{row['avg_mse']:<14.8f} {row['combined_rank']:<15.2f}")

    print("\n" + "=" * 80)
    print("TOP 5 WAVELETS BY METRIC")
    print("=" * 80)

    print("\n📊 Best Average SSIM (Structural Similarity):")
    top_ssim = df_clean.nlargest(5, 'avg_ssim')[['wavelet_type', 'avg_ssim']]
    for i, (idx, row) in enumerate(top_ssim.iterrows(), 1):
        print(f"  {i}. {row['wavelet_type']:<12} - {row['avg_ssim']:.6f}")

    print("\n📉 Best Average MSE (Mean Squared Error):")
    top_mse = df_clean.nsmallest(5, 'avg_mse')[['wavelet_type', 'avg_mse']]
    for i, (idx, row) in enumerate(top_mse.iterrows(), 1):
        print(f"  {i}. {row['wavelet_type']:<12} - {row['avg_mse']:.8f}")

    print("\n" + "=" * 80)
    print("MODALITY-SPECIFIC BEST PERFORMERS")
    print("=" * 80)

    for mod in modalities:
        ssim_col = f'eval/ssim_{mod}_mean'
        mse_col = f'eval/mse_{mod}_mean'
        
        best_ssim = df_clean.loc[df_clean[ssim_col].idxmax()]
        best_mse = df_clean.loc[df_clean[mse_col].idxmin()]
        
        print(f"\n{mod.upper()}:")
        print(f"  Best SSIM: {best_ssim['wavelet_type']:<12} ({best_ssim[ssim_col]:.6f})")
        print(f"  Best MSE:  {best_mse['wavelet_type']:<12} ({best_mse[mse_col]:.8f})")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    best_overall = df_ranked.iloc[0]
    print(f"\n✅ Overall Best Wavelet: {best_overall['wavelet_type'].upper()}")
    print(f"   - Average SSIM: {best_overall['avg_ssim']:.6f}")
    print(f"   - Average MSE:  {best_overall['avg_mse']:.8f}")
    print(f"   - Combined Rank Score: {best_overall['combined_rank']:.2f}")

    # Calculate performance spread
    ssim_range = df_clean['avg_ssim'].max() - df_clean['avg_ssim'].min()
    mse_range = df_clean['avg_mse'].max() - df_clean['avg_mse'].min()
    print(f"\n📈 Performance Variation:")
    print(f"   - SSIM range: {ssim_range:.6f} "
          f"({(ssim_range/df_clean['avg_ssim'].mean()*100):.2f}% of mean)")
    print(f"   - MSE range:  {mse_range:.8f} "
          f"({(mse_range/df_clean['avg_mse'].mean()*100):.2f}% of mean)")

    # Identify wavelet families
    families = {
        'Daubechies': [w for w in df_clean['wavelet_type'] if w.startswith('db')],
        'Symlets': [w for w in df_clean['wavelet_type'] if w.startswith('sym')],
        'Coiflets': [w for w in df_clean['wavelet_type'] if w.startswith('coif')],
        'Haar': [w for w in df_clean['wavelet_type'] if w == 'haar']
    }

    print(f"\n🔬 Wavelet Family Analysis:")
    for family, members in families.items():
        if members:
            family_df = df_clean[df_clean['wavelet_type'].isin(members)]
            avg_ssim = family_df['avg_ssim'].mean()
            avg_mse = family_df['avg_mse'].mean()
            print(f"   {family:<15} (n={len(members)}): "
                  f"Avg SSIM={avg_ssim:.6f}, Avg MSE={avg_mse:.8f}")

def main():
    """Main execution function"""
    print("=" * 80)
    print("Wavelet Performance Analysis")
    print("=" * 80)
    print()
    
    # Get input file
    input_file = get_input_file()
    
    # Load and clean data
    df_clean = load_and_clean_data(input_file)
    
    # Create visualizations
    df_clean = create_visualizations(df_clean, OUTPUT_VIZ)
    
    # Print analysis
    print_analysis(df_clean)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
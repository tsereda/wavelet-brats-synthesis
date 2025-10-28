#!/usr/bin/env python3
"""
Setup script for BraTS evaluation
- Installs dependencies
- Downloads checkpoints from W&B
- Validates data paths
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def install_dependencies():
    """Install required Python packages"""
    print("=" * 60)
    print("Installing Python dependencies...")
    print("=" * 60)
    
    packages = [
        "torch",
        "torchvision", 
        "nibabel",
        "numpy",
        "wandb",
        "monai",
        "PyWavelets",  # pywt
        "matplotlib",
        "pillow",
        "blobfile",
        "tqdm"
    ]
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--no-cache-dir", *packages
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def download_checkpoints_from_wandb(run_path, output_dir):
    """Download checkpoints from a W&B run"""
    print("=" * 60)
    print(f"Downloading checkpoints from W&B run: {run_path}")
    print("=" * 60)
    
    try:
        import wandb
        
        # Initialize API
        api = wandb.Api()
        
        # Get the run
        print(f"Fetching run metadata...")
        run = api.run(run_path)
        
        print(f"Run: {run.name}")
        print(f"State: {run.state}")
        print(f"Config modality: {run.config.get('contr', 'unknown')}")
        
        # Find checkpoint files
        checkpoint_files = [f for f in run.files() if f.name.endswith('.pt')]
        
        if not checkpoint_files:
            print("❌ No checkpoint files found in this run!")
            return False
        
        print(f"\nFound {len(checkpoint_files)} checkpoint files:")
        for f in checkpoint_files:
            print(f"  - {f.name}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download each checkpoint
        print(f"\nDownloading to {output_dir}...")
        for file in checkpoint_files:
            print(f"  Downloading {file.name}...", end=" ")
            file.download(root=output_dir, replace=True)
            print("✅")
        
        # List downloaded files
        downloaded = list(Path(output_dir).glob("*.pt"))
        print(f"\n✅ Downloaded {len(downloaded)} checkpoints:")
        for f in downloaded:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading checkpoints: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_checkpoints_from_url(url, output_dir):
    """Download and extract checkpoints from a URL"""
    print("=" * 60)
    print(f"Downloading checkpoints from URL: {url}")
    print("=" * 60)
    
    try:
        import urllib.request
        import zipfile
        import tarfile
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine file type from URL
        filename = url.split('/')[-1]
        local_path = f"/tmp/{filename}"
        
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"✅ Downloaded to {local_path}")
        
        # Extract based on file type
        print(f"Extracting...")
        if filename.endswith('.zip'):
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        elif filename.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(local_path, 'r:gz') as tar_ref:
                tar_ref.extractall(output_dir)
        else:
            print(f"⚠️  Unknown archive format: {filename}")
            return False
        
        # List extracted files
        extracted = list(Path(output_dir).glob("*.pt"))
        print(f"✅ Extracted {len(extracted)} checkpoints to {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading from URL: {e}")
        return False


def validate_data_paths(data_dir, checkpoint_dir):
    """Validate that required data exists"""
    print("=" * 60)
    print("Validating data paths...")
    print("=" * 60)
    
    issues = []
    
    # Check data directory
    if not os.path.exists(data_dir):
        issues.append(f"❌ Data directory not found: {data_dir}")
    else:
        # Count cases
        cases = [d for d in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('BraTS')]
        if len(cases) == 0:
            issues.append(f"❌ No BraTS cases found in {data_dir}")
        else:
            print(f"✅ Found {len(cases)} cases in {data_dir}")
            
            # Check first case has required files
            first_case = os.path.join(data_dir, cases[0])
            required_modalities = ['t1n', 't1c', 't2w', 't2f']
            case_files = os.listdir(first_case)
            
            for mod in required_modalities:
                if not any(mod in f for f in case_files):
                    issues.append(f"⚠️  Sample case missing {mod} modality")
    
    # Check checkpoint directory
    if not os.path.exists(checkpoint_dir):
        issues.append(f"❌ Checkpoint directory not found: {checkpoint_dir}")
    else:
        checkpoints = list(Path(checkpoint_dir).glob("*.pt"))
        if len(checkpoints) == 0:
            issues.append(f"❌ No checkpoints found in {checkpoint_dir}")
        else:
            print(f"✅ Found {len(checkpoints)} checkpoints")
            
            # Check for all modalities
            modalities = set()
            for ckpt in checkpoints:
                if 'brats_' in ckpt.name:
                    parts = ckpt.name.split('_')
                    if len(parts) >= 2:
                        modalities.add(parts[1])
            
            if modalities:
                print(f"   Modalities: {', '.join(sorted(modalities))}")
            
            missing_mods = set(['t1n', 't1c', 't2w', 't2f']) - modalities
            if missing_mods:
                issues.append(f"⚠️  Missing checkpoints for: {', '.join(missing_mods)}")
    
    # Print issues
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ All paths validated successfully")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Setup BraTS evaluation environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full setup with W&B checkpoint download
  python setup_evaluation.py --install-deps --wandb-run timgsereda/wavelet-brats-synthesis/abc123

  # Just install dependencies
  python setup_evaluation.py --install-deps

  # Download from URL
  python setup_evaluation.py --checkpoint-url https://example.com/checkpoints.zip

  # Validate existing setup
  python setup_evaluation.py --data-dir /data/validation --checkpoint-dir /data/checkpoints
        """
    )
    
    parser.add_argument('--install-deps', action='store_true',
                       help='Install Python dependencies')
    parser.add_argument('--wandb-run', type=str,
                       help='W&B run path (entity/project/run_id) to download checkpoints from')
    parser.add_argument('--checkpoint-url', type=str,
                       help='URL to download checkpoints from')
    parser.add_argument('--checkpoint-dir', type=str, default='/data/checkpoints',
                       help='Directory to save checkpoints (default: /data/checkpoints)')
    parser.add_argument('--data-dir', type=str, default='/data/validation_COMPLETE',
                       help='Directory containing validation data (default: /data/validation_COMPLETE)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate paths without downloading')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("BraTS Evaluation Setup")
    print("=" * 60 + "\n")
    
    success = True
    
    # Step 1: Install dependencies
    if args.install_deps:
        if not install_dependencies():
            print("\n❌ Failed to install dependencies")
            return 1
        print()
    
    # Step 2: Download checkpoints
    if args.wandb_run:
        if not download_checkpoints_from_wandb(args.wandb_run, args.checkpoint_dir):
            print("\n❌ Failed to download checkpoints from W&B")
            success = False
        print()
    
    if args.checkpoint_url:
        if not download_checkpoints_from_url(args.checkpoint_url, args.checkpoint_dir):
            print("\n❌ Failed to download checkpoints from URL")
            success = False
        print()
    
    # Step 3: Validate paths
    if not args.validate_only or (args.wandb_run or args.checkpoint_url):
        if not validate_data_paths(args.data_dir, args.checkpoint_dir):
            print("\n⚠️  Validation found issues, but continuing...")
            print("You can fix paths and re-run with --validate-only")
        print()
    
    if success:
        print("=" * 60)
        print("✅ Setup complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Verify paths are correct:")
        print(f"     Data: {args.data_dir}")
        print(f"     Checkpoints: {args.checkpoint_dir}")
        print("  2. Run evaluation:")
        print("     python app/scripts/complete_dataset.py \\")
        print(f"       --input_dir {args.data_dir} \\")
        print(f"       --checkpoint_dir {args.checkpoint_dir} \\")
        print("       --evaluation_mode --evaluate_metrics")
        return 0
    else:
        print("=" * 60)
        print("❌ Setup completed with errors")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
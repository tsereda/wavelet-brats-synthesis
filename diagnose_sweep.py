#!/usr/bin/env python3
"""
Diagnostic script to inspect W&B sweep runs and their artifacts
"""
import wandb

def diagnose_sweep(sweep_id, entity='timgsereda', project='wavelet-brats-synthesis-slice-prediction'):
    """Check what's available in sweep runs"""
    api = wandb.Api()
    
    print(f"Inspecting sweep: {entity}/{project}/{sweep_id}")
    print("="*80)
    
    try:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
        runs = sweep.runs
        
        print(f"\nFound {len(runs)} runs in sweep")
        print("="*80)
        
        for i, run in enumerate(runs[:10]):  # Check first 10 runs
            print(f"\n[Run {i+1}] {run.name} (ID: {run.id})")
            print(f"  State: {run.state}")
            print(f"  Config: model_type={run.config.get('model_type')}, wavelet={run.config.get('wavelet')}")
            
            # Check artifacts
            artifacts = list(run.logged_artifacts())
            print(f"  Artifacts: {len(artifacts)} total")
            
            for artifact in artifacts:
                print(f"    - {artifact.name} (type: {artifact.type})")
                print(f"      Aliases: {artifact.aliases}")
                if artifact.type == 'model':
                    print(f"      ✓ Model artifact found!")
            
            # Check files
            try:
                files = list(run.files())
                checkpoint_files = [f for f in files if '.pth' in f.name or '.pt' in f.name]
                if checkpoint_files:
                    print(f"  Checkpoint files: {len(checkpoint_files)}")
                    for f in checkpoint_files[:5]:
                        print(f"    - {f.name}")
            except Exception as e:
                print(f"  Error listing files: {e}")
        
        print("\n" + "="*80)
        print("SUMMARY:")
        print("="*80)
        
        # Count runs with model artifacts
        runs_with_artifacts = 0
        runs_with_checkpoints = 0
        
        for run in runs:
            artifacts = list(run.logged_artifacts())
            has_model = any(a.type == 'model' for a in artifacts)
            if has_model:
                runs_with_artifacts += 1
            
            try:
                files = list(run.files())
                has_checkpoint = any('.pth' in f.name or '.pt' in f.name for f in files)
                if has_checkpoint:
                    runs_with_checkpoints += 1
            except Exception as e:
                print(f"  Error listing files for run {run.id} in summary: {e}")
        
        print(f"Runs with model artifacts: {runs_with_artifacts}/{len(runs)}")
        print(f"Runs with checkpoint files: {runs_with_checkpoints}/{len(runs)}")
        
        if runs_with_artifacts == 0 and runs_with_checkpoints == 0:
            print("\n⚠️  NO CHECKPOINTS FOUND!")
            print("This means either:")
            print("  1. Training didn't save checkpoints properly")
            print("  2. Checkpoints weren't uploaded to W&B")
            print("  3. You need to use local checkpoint files instead")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    diagnose_sweep('5mfl25i8')
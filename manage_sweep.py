#!/usr/bin/env python3
"""
BraTS Training Management - Unified Script
Creates wandb sweep + deploys K8s training pods with dual checkpoint saving

Dual Checkpoint Saving:
  - Wandb: Automatic versioning, tracking, and cloud backup
  - PVC: Persistent storage, survives pod restarts, faster access

Usage:
    python manage_sweep.py                           # Create sweep + deploy 4 pods (DEFAULT)
    python manage_sweep.py --num-pods 6             # Deploy 6 pods
    python manage_sweep.py --save-iters 100,500,1000 # Save checkpoints at specific iterations
"""

import os
import sys
import yaml
import wandb
import subprocess
import argparse

# ============================================================================
# WANDB SWEEP MANAGEMENT
# ============================================================================

def load_sweep_config(config_path="sweep.yml"):
    """Load sweep configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: {config_path} not found!")
        print("Creating default sweep.yml...")
        create_default_sweep_config()
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


def create_default_sweep_config():
    """Create a default sweep.yml if it doesn't exist"""
    default_config = """project: fast-cwdm-brats
program: app/scripts/train.py
method: random
metric:
  name: val/ssim
  goal: maximize
parameters:
  data_dir:
    value: ./datasets/BRATS2023/training
  dataset:
    value: brats
  image_size:
    value: 224
  batch_size:
    value: 2
  num_workers:
    value: 12
  diffusion_steps:
    value: 100
  sample_schedule:
    value: sampled
  dims:
    value: 3
  num_channels:
    value: 64
  in_channels:
    value: 32
  out_channels:
    value: 8
  lr:
    values: [1e-5, 5e-5, 1e-4]
  save_interval:
    value: 500
  log_interval:
    value: 100
  contr:
    values: ['t1n', 't1c', 't2w', 't2f']
"""
    with open('sweep.yml', 'w') as f:
        f.write(default_config)
    print("✅ Created default sweep.yml")


def create_sweep(config_path="sweep.yml", entity=None, project=None):
    """Create a new wandb sweep and return sweep ID"""
    config = load_sweep_config(config_path)
    
    # Override entity/project if provided
    if entity:
        config['entity'] = entity
    if project:
        config['project'] = project
    
    entity = entity or config.get('entity', 'timgsereda')
    project = project or config.get('project', 'fast-cwdm-brats')
    
    print(f"\n📊 Creating W&B sweep in {entity}/{project}")
    
    try:
        # Create sweep
        sweep_id = wandb.sweep(config, entity=entity, project=project)
        
        print(f"✅ Sweep created: {sweep_id}")
        print(f"🔗 View at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
        
        return sweep_id
        
    except Exception as e:
        print(f"❌ Error creating sweep: {e}")
        return None


# ============================================================================
# KUBERNETES POD GENERATION
# ============================================================================

def generate_pod_yamls(sweep_id="", template_path="agent_pod_tr.yml", output_dir="k8s/training_pods", 
                       num_pods=3, save_iterations=None, pvc_name="brats2025-checkpoints"):
    """Generate numbered pod YAMLs from template with dual checkpoint saving"""
    
    # Read template
    try:
        with open(template_path, 'r') as f:
            template = f.read()
    except FileNotFoundError:
        print(f"❌ Error: Template file '{template_path}' not found!")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    for i in range(1, num_pods + 1):
        # Replace placeholders with pod number
        yaml_content = template.replace("{POD_NUM}", str(i))
        
        # Inject sweep ID if present
        if sweep_id:
            yaml_content = yaml_content.replace("{SWEEP_ID}", sweep_id)
        
        # Inject PVC name for checkpoint saving
        yaml_content = yaml_content.replace("{PVC_NAME}", pvc_name)
        
        # Inject save iterations if specified
        if save_iterations:
            save_iters_str = ",".join(map(str, save_iterations))
            yaml_content = yaml_content.replace("{SAVE_ITERATIONS}", save_iters_str)
        
        output_file = os.path.join(output_dir, f"wandb-sweep-agent-{i}.yml")
        
        with open(output_file, 'w') as f:
            f.write(yaml_content)
        
        generated_files.append(output_file)
        print(f"✅ Generated: {output_file}")
    
    return generated_files


def deploy_pods(pod_files):
    """Deploy pods to Kubernetes"""
    
    print(f"\n🚀 Deploying {len(pod_files)} training pods to Kubernetes...")
    
    for pod_file in pod_files:
        pod_name = os.path.basename(pod_file).replace('.yml', '')
        
        try:
            print(f"\n  Deploying {pod_name}...")
            result = subprocess.run(
                ['kubectl', 'apply', '-f', pod_file],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"  ✅ {pod_name}: {result.stdout.strip()}")
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {pod_name} failed: {e.stderr}")
        except FileNotFoundError:
            print(f"  ❌ kubectl not found. Please install kubectl or deploy manually:")
            print(f"     kubectl apply -f {pod_file}")
            return False
    
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BraTS Training Management - Create sweep + deploy training pods with dual checkpoint saving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Create sweep + deploy 4 pods
  python manage_sweep.py
  
  # Custom number of pods
  python manage_sweep.py --num-pods 6
  
  # Save checkpoints at specific iterations
  python manage_sweep.py --save-iters 100,500,1000,2000
  
  # Specify custom PVC for checkpoint storage
  python manage_sweep.py --pvc-name my-checkpoints-pvc
  
Checkpoint Saving:
  - Checkpoints are saved to BOTH wandb and the PVC
  - PVC checkpoints persist across pod restarts
  - Wandb checkpoints are versioned and tracked
  
Tip: Use 'wandb sweep --stop <sweep-id>' to cancel a sweep
        """
    )
    
    parser.add_argument('--entity', type=str, default='timgsereda',
                       help='W&B entity (default: timgsereda)')
    parser.add_argument('--project', type=str, default='wavelet-brats-synthesis',
                       help='W&B project (default: wavelet-brats-synthesis)')
    parser.add_argument('--num-pods', type=int, default=4,
                       help='Number of sweep agent pods to deploy (default: 4)')
    parser.add_argument('--template', type=str, default='agent_pod_tr.yml',
                       help='Path to pod template YAML (default: agent_pod_tr.yml)')
    parser.add_argument('--save-iters', type=str, default=None,
                       help='Comma-separated list of iterations to save checkpoints (e.g., "100,500,1000")')
    parser.add_argument('--pvc-name', type=str, default='brats2025-checkpoints',
                       help='Name of PVC for checkpoint storage (default: brats2025-checkpoints)')
    
    args = parser.parse_args()
    
    # Parse save iterations
    save_iterations = None
    if args.save_iters:
        try:
            save_iterations = [int(x.strip()) for x in args.save_iters.split(',')]
            print(f"\n💾 Checkpoints will be saved at iterations: {save_iterations}")
        except ValueError:
            print(f"❌ Error: Invalid save-iters format. Use comma-separated integers.")
            sys.exit(1)
    
    print("=" * 60)
    print("🧠 BraTS Training Management")
    print("=" * 60)
    
    # Create sweep
    print("\n[Step 1/3] Creating W&B sweep...")
    sweep_id = create_sweep(entity=args.entity, project=args.project)
    
    if not sweep_id:
        print("❌ Failed to create sweep. Aborting...")
        sys.exit(1)
    
    # Generate pod YAMLs
    print(f"\n[Step 2/3] Generating {args.num_pods} Kubernetes pod YAMLs...")
    pod_files = generate_pod_yamls(
        sweep_id=sweep_id,
        template_path=args.template,
        num_pods=args.num_pods,
        save_iterations=save_iterations,
        pvc_name=args.pvc_name
    )
    
    print(f"\n📁 Pod YAMLs saved to: k8s/training_pods/")
    
    # Deploy pods
    print("\n[Step 3/3] Deploying pods to Kubernetes...")
    
    deploy_pods(pod_files)
    
    print("\n" + "=" * 60)
    print("✨ All done!")
    print("=" * 60)
    
    print(f"\n📊 Sweep ID: {sweep_id}")
    print(f"🔗 View at: https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}")
    
    print(f"\n💾 Checkpoint Configuration:")
    print(f"  - Dual saving enabled: wandb + PVC")
    print(f"  - PVC name: {args.pvc_name}")
    if save_iterations:
        print(f"  - Save iterations: {save_iterations}")
    else:
        print(f"  - Save interval: Using sweep config default")
    
    print(f"\n🎯 Monitor pods:")
    print(f"  kubectl get pods -l app=wandb-sweep")
    print(f"\n📋 Check logs:")
    for i in range(1, min(args.num_pods + 1, 4)):
        print(f"  kubectl logs -f wandb-sweep-agent-{i}")
    if args.num_pods > 3:
        print(f"  ... (and {args.num_pods - 3} more)")
    
    print(f"\n💾 Access checkpoints:")
    print(f"  kubectl exec -it wandb-sweep-agent-1 -- ls /data/checkpoints")
    
    print(f"\n🛑 Stop all pods:")
    print(f"  kubectl delete pods -l app=wandb-sweep")


if __name__ == "__main__":
    main()
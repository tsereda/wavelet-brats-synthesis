#!/usr/bin/env python3
"""
BraTS Training Management - Unified Script
Default: Creates wandb sweep + deploys 4 K8s training pods

Usage:
    python manage_training.py              # Create sweep + deploy 4 pods (DEFAULT)
    python manage_training.py --job        # Create sweep + deploy 4 jobs
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
  name: val/mse
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

def generate_pod_yamls(sweep_id="", template_path="agent_pod_tr.yml", output_dir="k8s/training_pods", num_pods=3):
    """Generate numbered pod/job YAMLs from template"""
    
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
        
        # Also inject sweep ID if present
        if sweep_id:
            yaml_content = yaml_content.replace("{SWEEP_ID}", sweep_id)
        
        # Determine if it's a job or pod based on template path
        if "job" in template_path.lower():
            output_file = os.path.join(output_dir, f"wandb-sweep-job-{i}.yml")
        else:
            output_file = os.path.join(output_dir, f"wandb-sweep-agent-{i}.yml")
        
        with open(output_file, 'w') as f:
            f.write(yaml_content)
        
        generated_files.append(output_file)
        print(f"✅ Generated: {output_file}")
    
    return generated_files


def deploy_pods(pod_files):
    """Deploy pods/jobs to Kubernetes"""
    
    print(f"\n🚀 Deploying {len(pod_files)} training agents to Kubernetes...")
    
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
        description="BraTS Training Management - Create sweep + deploy training pods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Create sweep + deploy 4 pods
  python manage_training.py
  
  # Create sweep + deploy jobs
  python manage_training.py --job
  
  # Custom number of pods/jobs
  python manage_training.py --num-agents 5
  
Tip: Use 'wandb sweep --stop <sweep-id>' to cancel a sweep
        """
    )
    
    parser.add_argument('--entity', type=str, default='timgsereda',
                       help='W&B entity (default: timgsereda)')
    parser.add_argument('--project', type=str, default='wavelet-brats-synthesis',
                       help='W&B project (default: wavelet-brats-synthesis)')
    parser.add_argument('--num-agents', type=int, default=4,
                       help='Number of sweep agents to deploy (default: 4)')
    parser.add_argument('--job', action='store_true',
                       help='Deploy as Kubernetes Jobs instead of Pods')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 BraTS Training Management")
    print("=" * 60)
    
    # Create sweep
    print("\n[Step 1/3] Creating W&B sweep...")
    sweep_id = create_sweep(entity=args.entity, project=args.project)
    
    if not sweep_id:
        print("❌ Failed to create sweep. Aborting...")
        sys.exit(1)
    
    # Determine template and deployment type
    if args.job:
        template_path = 'agent_job_tr.yml'
        deployment_type = 'Jobs'
    else:
        template_path = 'agent_pod_tr.yml'
        deployment_type = 'Pods'
    
    # Generate YAMLs
    print(f"\n[Step 2/3] Generating {args.num_agents} Kubernetes {deployment_type.lower()} YAMLs...")
    pod_files = generate_pod_yamls(
        sweep_id=sweep_id,
        template_path=template_path,
        num_pods=args.num_agents
    )
    
    print(f"\n📁 {deployment_type} YAMLs saved to: k8s/training_pods/")
    
    # Deploy 
    print(f"\n[Step 3/3] Deploying {deployment_type.lower()} to Kubernetes...")
    
    deploy_pods(pod_files)
    
    print("\n" + "=" * 60)
    print("✨ All done!")
    print("=" * 60)
    
    print(f"\n📊 Sweep ID: {sweep_id}")
    print(f"🔗 View at: https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}")
    
    if args.job:
        print(f"\n🎯 Monitor jobs:")
        print(f"  kubectl get jobs -l app=wandb-sweep")
        print(f"\n📋 Check logs:")
        for i in range(1, args.num_agents + 1):
            print(f"  kubectl logs -f job/wandb-sweep-agent-{i}")
        print(f"\n🛑 Stop all jobs:")
        print(f"  kubectl delete jobs -l app=wandb-sweep")
    else:
        print(f"\n🎯 Monitor pods:")
        print(f"  kubectl get pods -l app=wandb-sweep")
        print(f"\n📋 Check logs:")
        for i in range(1, args.num_agents + 1):
            print(f"  kubectl logs -f wandb-sweep-agent-{i}")
        print(f"\n🛑 Stop all pods:")
        print(f"  kubectl delete pods -l app=wandb-sweep")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Wandb Sweep Management Script
Easily create and cancel sweeps
"""

import argparse
import sys
import yaml
import wandb


def load_sweep_config(config_path="sweep.yml"):
    """Load sweep configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_sweep(config_path="sweep.yml", entity=None, project=None):
    """Create a new sweep and return sweep ID"""
    config = load_sweep_config(config_path)
    
    # Override entity/project if provided
    if entity:
        config['entity'] = entity
    if project:
        config['project'] = project
    
    print(f"📊 Creating sweep in {config.get('entity', 'default')}/{config['project']}")
    
    # Create sweep
    sweep_id = wandb.sweep(config, entity=entity, project=project)
    
    print(f"✅ Sweep created: {sweep_id}")
    print(f"🔗 View at: https://wandb.ai/{entity or wandb.api.default_entity}/{project or config['project']}/sweeps/{sweep_id}")
    print(f"\n🚀 To launch agents, run:")
    print(f"   wandb agent {entity or wandb.api.default_entity}/{project or config['project']}/{sweep_id}")
    
    return sweep_id


def cancel_sweep(sweep_id, entity=None, project=None):
    """Cancel a running sweep"""
    api = wandb.Api()
    
    # Parse sweep path
    if '/' in sweep_id:
        sweep_path = sweep_id
    else:
        entity = entity or wandb.api.default_entity
        project = project or "wavelet-brats-synthesis"
        sweep_path = f"{entity}/{project}/{sweep_id}"
    
    print(f"🛑 Canceling sweep: {sweep_path}")
    
    try:
        sweep = api.sweep(sweep_path)
        
        # Stop the sweep
        sweep.stop()
        
        print(f"✅ Sweep {sweep_id} stopped successfully")
        
    except Exception as e:
        print(f"❌ Error stopping sweep: {e}")
        return False
    
    return True


def list_sweeps(entity=None, project=None):
    """List recent sweeps"""
    api = wandb.Api()
    
    entity = entity or wandb.api.default_entity
    project = project or "wavelet-brats-synthesis"
    
    print(f"📋 Recent sweeps in {entity}/{project}:\n")
    
    try:
        sweeps = api.project(f"{entity}/{project}").sweeps()
        
        for i, sweep in enumerate(sweeps[:10], 1):
            state = sweep.state
            config = sweep.config
            
            # Get metric info
            metric_name = config.get('metric', {}).get('name', 'unknown')
            
            print(f"{i}. {sweep.id}")
            print(f"   State: {state}")
            print(f"   Metric: {metric_name}")
            print(f"   Method: {config.get('method', 'unknown')}")
            print(f"   Runs: {len(sweep.runs)}")
            print(f"   URL: {sweep.url}")
            print()
            
    except Exception as e:
        print(f"❌ Error listing sweeps: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Wandb Sweeps for BraTS Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new sweep
  python manage_sweep.py create
  
  # Create with custom entity/project
  python manage_sweep.py create --entity myteam --project myproject
  
  # Cancel a running sweep
  python manage_sweep.py cancel <sweep_id>
  
  # List recent sweeps
  python manage_sweep.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new sweep')
    create_parser.add_argument('--config', default='sweep.yml', help='Sweep config file')
    create_parser.add_argument('--entity', help='Wandb entity')
    create_parser.add_argument('--project', help='Wandb project')
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel a sweep')
    cancel_parser.add_argument('sweep_id', help='Sweep ID to cancel')
    cancel_parser.add_argument('--entity', help='Wandb entity')
    cancel_parser.add_argument('--project', help='Wandb project')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List recent sweeps')
    list_parser.add_argument('--entity', help='Wandb entity')
    list_parser.add_argument('--project', help='Wandb project')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'create':
        create_sweep(
            config_path=args.config,
            entity=args.entity,
            project=args.project
        )
    
    elif args.command == 'cancel':
        cancel_sweep(
            args.sweep_id,
            entity=args.entity,
            project=args.project
        )
    
    elif args.command == 'list':
        list_sweeps(
            entity=args.entity,
            project=args.project
        )


if __name__ == "__main__":
    main()
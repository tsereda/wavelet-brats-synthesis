#!/usr/bin/env python3
"""
Test script to validate the train/val split implementation.
Checks that the changes are syntactically correct and logically sound.
"""

import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_imports():
    """Test that all imports work"""
    try:
        from guided_diffusion.train_util import TrainLoop, DirectRegressionLoop
        print("✅ Imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_train_loop_has_validation():
    """Test that TrainLoop has run_validation method"""
    try:
        from guided_diffusion.train_util import TrainLoop
        assert hasattr(TrainLoop, 'run_validation'), "TrainLoop missing run_validation method"
        print("✅ TrainLoop has run_validation method")
        return True
    except Exception as e:
        print(f"❌ TrainLoop validation check failed: {e}")
        return False

def test_direct_loop_has_validation():
    """Test that DirectRegressionLoop has run_validation method"""
    try:
        from guided_diffusion.train_util import DirectRegressionLoop
        assert hasattr(DirectRegressionLoop, 'run_validation'), "DirectRegressionLoop missing run_validation method"
        print("✅ DirectRegressionLoop has run_validation method")
        return True
    except Exception as e:
        print(f"❌ DirectRegressionLoop validation check failed: {e}")
        return False

def test_train_script_syntax():
    """Test that train.py has valid syntax"""
    try:
        with open('app/scripts/train.py', 'r') as f:
            code = f.read()
        compile(code, 'app/scripts/train.py', 'exec')
        print("✅ train.py syntax valid")
        return True
    except SyntaxError as e:
        print(f"❌ train.py syntax error: {e}")
        return False

def check_wandb_keys():
    """Check that W&B logging keys have been updated"""
    try:
        with open('app/guided_diffusion/train_util.py', 'r') as f:
            content = f.read()
        
        # Check for new keys
        has_train_mse = 'train/mse' in content
        has_val_mse = 'val/mse' in content
        has_val_loss = 'val/loss' in content
        
        # Check that old keys are mostly removed (some may be in comments)
        old_metrics_count = content.count("'metrics/MSE'")
        
        print(f"✅ W&B key checks:")
        print(f"   - 'train/mse' present: {has_train_mse}")
        print(f"   - 'val/mse' present: {has_val_mse}")
        print(f"   - 'val/loss' present: {has_val_loss}")
        print(f"   - Old 'metrics/MSE' count: {old_metrics_count}")
        
        return has_train_mse and has_val_mse and has_val_loss
    except Exception as e:
        print(f"❌ W&B key check failed: {e}")
        return False

def check_train_val_split():
    """Check that train.py implements train/val split"""
    try:
        with open('app/scripts/train.py', 'r') as f:
            content = f.read()
        
        has_random_split = 'random_split' in content
        has_train_ds = 'train_ds' in content
        has_val_ds = 'val_ds' in content
        has_train_loader = 'train_loader' in content
        has_val_loader = 'val_loader' in content
        has_val_data_param = 'val_data=val_loader' in content
        
        print(f"✅ Train/val split checks:")
        print(f"   - Uses random_split: {has_random_split}")
        print(f"   - Creates train_ds: {has_train_ds}")
        print(f"   - Creates val_ds: {has_val_ds}")
        print(f"   - Creates train_loader: {has_train_loader}")
        print(f"   - Creates val_loader: {has_val_loader}")
        print(f"   - Passes val_data to TrainLoop: {has_val_data_param}")
        
        return all([has_random_split, has_train_ds, has_val_ds, 
                   has_train_loader, has_val_loader, has_val_data_param])
    except Exception as e:
        print(f"❌ Train/val split check failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Testing Validation Implementation")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Train.py Syntax", test_train_script_syntax),
        ("TrainLoop Validation", test_train_loop_has_validation),
        ("DirectRegressionLoop Validation", test_direct_loop_has_validation),
        ("W&B Logging Keys", check_wandb_keys),
        ("Train/Val Split", check_train_val_split)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        result = test_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())

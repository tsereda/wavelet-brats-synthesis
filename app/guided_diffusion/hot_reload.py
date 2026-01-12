"""
Hot-reload utility for development without interrupting training.
Allows updating debug/logging code during training runs.
"""
import sys
import importlib
from pathlib import Path


def reload_modules(module_names):
    """
    Reload specified modules to pick up code changes.
    
    Usage in train loop:
        from guided_diffusion.hot_reload import reload_modules
        if self.step % 100 == 0:  # Check every 100 steps
            reload_modules(['guided_diffusion.logger'])
    
    Args:
        module_names: List of module names to reload (e.g., ['guided_diffusion.logger'])
    """
    reloaded = []
    for name in module_names:
        if name in sys.modules:
            try:
                importlib.reload(sys.modules[name])
                reloaded.append(name)
            except Exception as e:
                print(f"⚠️  Hot-reload failed for {name}: {e}")
        else:
            print(f"⚠️  Module {name} not loaded yet, skipping reload")
    
    if reloaded:
        print(f"🔥 Hot-reloaded: {', '.join(reloaded)}")
    
    return reloaded


def check_and_reload_if_changed(module_names, last_mtimes=None):
    """
    Check if source files changed and reload if needed.
    
    Usage:
        from guided_diffusion.hot_reload import check_and_reload_if_changed
        self.reload_mtimes = {}  # In __init__
        
        if self.step % 100 == 0:
            check_and_reload_if_changed(
                ['guided_diffusion.logger', 'guided_diffusion.train_util'],
                self.reload_mtimes
            )
    
    Args:
        module_names: List of module names to watch
        last_mtimes: Dict to track last modification times (mutated in-place)
    
    Returns:
        List of reloaded modules
    """
    if last_mtimes is None:
        last_mtimes = {}
    
    reloaded = []
    for name in module_names:
        if name not in sys.modules:
            continue
        
        module = sys.modules[name]
        if not hasattr(module, '__file__') or module.__file__ is None:
            continue
        
        filepath = Path(module.__file__)
        if not filepath.exists():
            continue
        
        current_mtime = filepath.stat().st_mtime
        last_mtime = last_mtimes.get(name)
        
        if last_mtime is None:
            # First check, just record
            last_mtimes[name] = current_mtime
        elif current_mtime > last_mtime:
            # File changed, reload
            try:
                importlib.reload(module)
                last_mtimes[name] = current_mtime
                reloaded.append(name)
                print(f"🔥 Hot-reloaded {name} (file changed)")
            except Exception as e:
                print(f"⚠️  Hot-reload failed for {name}: {e}")
    
    return reloaded

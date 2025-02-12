import torch
import json
from pathlib import Path
import time
from datetime import datetime

def get_device(device=None):
    """Get the appropriate device, defaulting to MPS if available."""
    if device is not None:
        return device
    
    # Always try MPS first
    if torch.backends.mps.is_available():
        try:
            mps_device = torch.device("mps")
            # Quick test to ensure MPS is working
            test_tensor = torch.ones(1, device=mps_device)
            _ = test_tensor + 1
            return mps_device
        except:
            print("Warning: MPS available but not working properly, falling back to CPU")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def to_device(tensor_or_module, device=None):
    """Helper to move tensor or module to appropriate device."""
    target_device = get_device(device)
    if isinstance(tensor_or_module, (torch.Tensor, torch.nn.Module)):
        return tensor_or_module.to(target_device)
    return tensor_or_module 

def check_mps_compatibility():
    """Check if MPS is available and working correctly."""
    if not torch.backends.mps.is_available():
        return False
    
    try:
        # Test basic tensor operations on MPS
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        y = x + 1
        return True
    except:
        return False

def get_optimal_device():
    """Get the best available device with fallback options."""
    if check_mps_compatibility():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu") 

def verify_mps_tensor_operations(tensor):
    """Verify tensor operations and dtypes for MPS."""
    if not tensor.device.type == 'mps':
        return True
        
    try:
        # Test operations with dtype checking
        if tensor.dtype not in [torch.float32, torch.float16, torch.int32]:
            print(f"Warning: MPS may not fully support dtype {tensor.dtype}")
            
        result = tensor + 1
        result = tensor * 2
        result = torch.mean(tensor)
        return True
    except Exception as e:
        print(f"MPS tensor operation failed: {str(e)}")
        return False

def safe_to_device(tensor, device=None, dtype=None):
    """Safely move tensor to device with appropriate dtype."""
    target_device = get_device(device)
    try:
        # Always prefer MPS if available
        if target_device.type == 'mps':
            # Handle MPS-specific dtypes
            if dtype is None:
                if tensor.dtype == torch.float64:
                    dtype = torch.float32
                elif tensor.dtype == torch.int64:
                    dtype = torch.int32
            
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
                
        return tensor.to(target_device)
    except Exception as e:
        print(f"Warning: Device transfer failed: {str(e)}, falling back to CPU")
        return tensor.to('cpu') 

class CheckpointManager:
    def __init__(
        self,
        save_dir: str = './checkpoints',
        max_checkpoints: int = 5,
        save_freq: int = 1000  # Save every 1000 steps
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_freq = save_freq
        
        # Load checkpoint metadata if exists
        self.metadata_path = self.save_dir / 'checkpoint_metadata.json'
        self.metadata = self._load_metadata()
        
    def _load_metadata(self):
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {'checkpoints': [], 'last_step': 0}
    
    def save_checkpoint(self, model, optimizer, step, metrics=None):
        if step % self.save_freq != 0:
            return
            
        # Create checkpoint directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = self.save_dir / f'checkpoint_{step}_{timestamp}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and optimizer state
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'metrics': metrics or {},
            'timestamp': timestamp,
            'config': model.config if hasattr(model, 'config') else None
        }
        
        checkpoint_path = checkpoint_dir / 'checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Update 'latest' symbolic link
        latest_dir = self.save_dir / 'latest'
        if latest_dir.exists():
            latest_dir.unlink()
        latest_dir.symlink_to(checkpoint_dir.relative_to(self.save_dir))
        
        # Update metadata
        self.metadata['checkpoints'].append(str(checkpoint_dir))
        self.metadata['last_step'] = step
        
        # Remove old checkpoints if exceeding max_checkpoints
        if len(self.metadata['checkpoints']) > self.max_checkpoints:
            old_checkpoint = Path(self.metadata['checkpoints'].pop(0))
            if old_checkpoint.exists() and old_checkpoint != checkpoint_dir:
                import shutil
                shutil.rmtree(old_checkpoint)
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def load_latest_checkpoint(self, model, optimizer=None):
        if not self.metadata['checkpoints']:
            return None, 0
            
        latest_checkpoint = Path(self.metadata['checkpoints'][-1]) / 'checkpoint.pt'
        if not latest_checkpoint.exists():
            return None, 0
            
        checkpoint = torch.load(latest_checkpoint, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint['metrics'], checkpoint['step'] 
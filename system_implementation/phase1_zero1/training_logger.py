"""
Training logger that saves all metrics to JSON files.
Creates one JSON file per training run with timestamps.
"""

import json
import os
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    """
    Logs training metrics to JSON file.
    
    Usage:
        logger = TrainingLogger(out_dir='out', config=config)
        logger.log_iter(iter_num, loss, time, mfu)
        logger.log_eval(iter_num, train_loss, val_loss)
        logger.save()
    """
    
    def __init__(self, out_dir='out', config=None, run_name=None):
        """
        Initialize logger.
        
        Args:
            out_dir: Directory to save logs
            config: Training configuration dict
            run_name: Optional custom run name (default: timestamp)
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        
        # Generate unique filename with timestamp
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
        
        self.run_name = run_name
        self.log_file = self.out_dir / f"{run_name}.json"
        
        # Initialize log data structure
        self.data = {
            'run_name': run_name,
            'start_time': datetime.now().isoformat(),
            'config': config if config is not None else {},
            'training_iterations': [],
            'eval_steps': [],
            'checkpoints': [],
            'metadata': {}
        }
        
        print(f"Training logger initialized: {self.log_file}")
    
    def log_iter(self, iter_num, loss, time_ms, mfu=-100.0):
        """
        Log a training iteration.
        
        Args:
            iter_num: Iteration number
            loss: Training loss
            time_ms: Time in milliseconds
            mfu: Model FLOPs Utilization (percentage)
        """
        # Convert all values to native Python types
        import torch
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if isinstance(time_ms, torch.Tensor):
            time_ms = time_ms.item()
        if isinstance(mfu, torch.Tensor):
            mfu = mfu.item()
        
        self.data['training_iterations'].append({
            'iter': int(iter_num),
            'loss': float(loss),
            'time_ms': float(time_ms),
            'mfu': float(mfu)
        })
    
    def log_eval(self, iter_num, train_loss, val_loss, lr=None):
        """
        Log an evaluation step.
        
        Args:
            iter_num: Iteration number
            train_loss: Training loss
            val_loss: Validation loss
            lr: Learning rate (optional)
        """
        # Convert all values to native Python types
        import torch
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.item()
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()
        if lr is not None and isinstance(lr, torch.Tensor):
            lr = lr.item()
        
        eval_data = {
            'iter': int(iter_num),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'timestamp': datetime.now().isoformat()
        }
        if lr is not None:
            eval_data['lr'] = float(lr)
        
        self.data['eval_steps'].append(eval_data)
    
    def log_checkpoint(self, iter_num, val_loss, checkpoint_path):
        """
        Log a checkpoint save.
        
        Args:
            iter_num: Iteration number
            val_loss: Validation loss
            checkpoint_path: Path to checkpoint file
        """
        # Convert all values to native Python types
        import torch
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()
        
        self.data['checkpoints'].append({
            'iter': int(iter_num),
            'val_loss': float(val_loss),
            'path': str(checkpoint_path),
            'timestamp': datetime.now().isoformat()
        })
    
    def log_metadata(self, key, value):
        """
        Log arbitrary metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.data['metadata'][key] = value
    
    def finalize(self, final_iter=None, best_val_loss=None):
        """
        Finalize the log with summary statistics.
        
        Args:
            final_iter: Final iteration number
            best_val_loss: Best validation loss achieved
        """
        self.data['end_time'] = datetime.now().isoformat()
        
        # Compute summary statistics
        if self.data['training_iterations']:
            losses = [x['loss'] for x in self.data['training_iterations']]
            times = [x['time_ms'] for x in self.data['training_iterations']]
            mfus = [x['mfu'] for x in self.data['training_iterations'] if x['mfu'] > 0]
            
            self.data['summary'] = {
                'total_iterations': len(self.data['training_iterations']),
                'final_iter': final_iter if final_iter is not None else len(self.data['training_iterations']) - 1,
                'final_train_loss': losses[-1] if losses else None,
                'best_val_loss': best_val_loss,
                'avg_time_ms': sum(times) / len(times) if times else None,
                'avg_mfu': sum(mfus) / len(mfus) if mfus else None,
                'total_eval_steps': len(self.data['eval_steps']),
                'total_checkpoints': len(self.data['checkpoints'])
            }
    
    def _convert_to_serializable(self, obj):
        """Convert PyTorch tensors and other non-serializable objects to JSON-safe types."""
        import torch
        import numpy as np
        
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def save(self):
        """Save the log to JSON file."""
        # Convert all data to JSON-serializable format
        serializable_data = self._convert_to_serializable(self.data)
        
        with open(self.log_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def auto_save(self, interval=10):
        """
        Check if we should auto-save based on iteration count.
        
        Args:
            interval: Save every N iterations
        
        Returns:
            bool: True if saved, False otherwise
        """
        if len(self.data['training_iterations']) % interval == 0:
            self.save()
            return True
        return False


def load_training_log(log_file):
    """
    Load a training log from JSON file.
    
    Args:
        log_file: Path to JSON log file
    
    Returns:
        dict: Training log data
    """
    with open(log_file, 'r') as f:
        return json.load(f)


def compare_runs(log_files):
    """
    Compare multiple training runs.
    
    Args:
        log_files: List of JSON log file paths
    
    Returns:
        dict: Comparison data
    """
    runs = []
    for log_file in log_files:
        data = load_training_log(log_file)
        runs.append({
            'run_name': data['run_name'],
            'final_loss': data['summary']['final_train_loss'],
            'best_val_loss': data['summary']['best_val_loss'],
            'avg_mfu': data['summary']['avg_mfu'],
            'total_iters': data['summary']['total_iterations']
        })
    
    return {'runs': runs}


if __name__ == '__main__':
    # Example usage
    logger = TrainingLogger(out_dir='logs', config={'lr': 6e-4, 'batch_size': 12})
    
    # Simulate training
    for i in range(10):
        logger.log_iter(i, 10.0 - i * 0.5, 4000.0, 33.0)
        
        if i % 5 == 0:
            logger.log_eval(i, 10.0 - i * 0.5, 10.1 - i * 0.5)
    
    logger.finalize(final_iter=9, best_val_loss=5.1)
    logger.save()
    
    print(f"Log saved to: {logger.log_file}")
    
    # Load and display
    loaded = load_training_log(logger.log_file)
    print(f"\nSummary: {json.dumps(loaded['summary'], indent=2)}")


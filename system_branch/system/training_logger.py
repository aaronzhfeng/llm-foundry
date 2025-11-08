"""
Enhanced Training Logger for LLM Training
==========================================

Saves detailed training metrics to JSON files with comprehensive breakdowns:
- MFU component breakdown (hardware flops, tokens/s, etc.)
- Memory statistics (allocated, peak, reserved)
- Gradient statistics (norms, values distribution)
- Per-iteration and per-evaluation metrics

Creates one JSON file per training run with timestamps.
"""

import json
import os
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    """
    Logs training metrics to JSON file with detailed MFU/memory/gradient tracking.
    
    Usage:
        logger = TrainingLogger(out_dir='out', config=config)
        logger.log_iter_detailed(iter_num, loss, dt_ms, mfu_breakdown, memory_stats, grad_stats)
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
            'startup_info': {},
            'training_iterations': [],
            'eval_steps': [],
            'checkpoints': [],
            'metadata': {}
        }
        
        print(f"Training logger initialized: {self.log_file}")
    
    def log_iter(self, iter_num, loss, time_ms, mfu=-100.0):
        """
        Log a training iteration (simple version for compatibility).
        
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
    
    def log_iter_detailed(self, iter_num, loss, dt_ms, mfu_breakdown, memory_stats=None, grad_stats=None):
        """
        Log a training iteration with detailed metrics.
        
        Args:
            iter_num: Iteration number
            loss: Training loss
            dt_ms: Time in milliseconds
            mfu_breakdown: Dict from estimate_mfu_detailed()
            memory_stats: Dict from get_memory_stats()
            grad_stats: Dict from get_gradient_stats()
        """
        import torch
        
        # Convert to native Python types
        def to_native(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [to_native(item) for item in obj]
            else:
                return float(obj) if isinstance(obj, (int, float)) else obj
        
        log_entry = {
            'iter': int(iter_num),
            'loss': float(loss),
            'time_ms': float(dt_ms),
            'mfu': to_native(mfu_breakdown),
        }
        
        if memory_stats:
            log_entry['memory'] = to_native(memory_stats)
        
        if grad_stats:
            log_entry['gradients'] = to_native(grad_stats)
        
        self.data['training_iterations'].append(log_entry)
    
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
    
    def log_startup_info(self, model, optimizer, config, hardware_info=None):
        """
        Log detailed startup information.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            config: Training configuration dict
            hardware_info: Optional hardware information dict
        """
        import torch
        
        startup_info = {
            'timestamp': datetime.now().isoformat(),
            'model': {
                'total_params': sum(p.numel() for p in model.parameters()),
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'non_embedding_params': model.get_num_params() if hasattr(model, 'get_num_params') else None,
            },
            'optimizer': {
                'type': type(optimizer).__name__,
                'param_groups': len(optimizer.param_groups),
            },
            'config': config,
        }
        
        if hardware_info:
            startup_info['hardware'] = hardware_info
        
        self.data['startup_info'] = startup_info
    
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
            
            # Try to get MFU values (handle both simple and detailed formats)
            mfus = []
            for x in self.data['training_iterations']:
                if isinstance(x.get('mfu'), dict):
                    mfu_val = x['mfu'].get('mfu_percent', 0)
                else:
                    mfu_val = x.get('mfu', 0)
                if mfu_val > 0:
                    mfus.append(mfu_val)
            
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
    
    # Simulate training with detailed metrics
    for i in range(10):
        mfu_breakdown = {
            'mfu_percent': 30.0 + i,
            'achieved_tflops': 100.0 + i * 2,
            'hardware_peak_tflops': 312.0,
            'tokens_per_sec': 3500 + i * 10,
            'flops_per_token': 28.5e9,
        }
        memory_stats = {
            'allocated_gb': 12.0 + i * 0.1,
            'max_allocated_gb': 15.0 + i * 0.1,
        }
        
        logger.log_iter_detailed(i, 10.0 - i * 0.5, 4000.0, mfu_breakdown, memory_stats)
        
        if i % 5 == 0:
            logger.log_eval(i, 10.0 - i * 0.5, 10.1 - i * 0.5)
    
    logger.finalize(final_iter=9, best_val_loss=5.1)
    logger.save()
    
    print(f"Log saved to: {logger.log_file}")
    
    # Load and display
    loaded = load_training_log(logger.log_file)
    print(f"\nSummary: {json.dumps(loaded['summary'], indent=2)}")


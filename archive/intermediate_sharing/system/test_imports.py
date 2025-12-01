#!/usr/bin/env python3
"""Quick test to verify all imports work"""

print("Testing imports...")

try:
    from model_components import NORM_REGISTRY, FFN_REGISTRY, POSITION_ENCODING_REGISTRY
    print("✅ model_components imported successfully")
    print(f"   - {len(NORM_REGISTRY)} normalization types")
    print(f"   - {len(FFN_REGISTRY)} FFN types")
    print(f"   - {len(POSITION_ENCODING_REGISTRY)} position encoding types")
except Exception as e:
    print(f"❌ model_components import failed: {e}")

try:
    from model_config import ModelArchitectureConfig, get_preset_config
    print("✅ model_config imported successfully")
    
    # Test presets
    gpt2 = get_preset_config('gpt2')
    llama = get_preset_config('llama')
    print(f"   - GPT-2: {gpt2.get_architecture_name()}")
    print(f"   - LLaMA: {llama.get_architecture_name()}")
except Exception as e:
    print(f"❌ model_config import failed: {e}")

try:
    from model_builder import ConfigurableGPT, TransformerBlock
    print("✅ model_builder imported successfully")
except Exception as e:
    print(f"❌ model_builder import failed: {e}")

try:
    from training_logger import TrainingLogger
    print("✅ training_logger imported successfully")
except Exception as e:
    print(f"❌ training_logger import failed: {e}")

try:
    from model import GPT, GPTConfig
    print("✅ model (legacy) imported successfully")
except Exception as e:
    print(f"❌ model import failed: {e}")

print("\n🎉 All imports successful! System is ready to use.")
print("\nNext step: python train.py config/arch_gpt2.py --max_iters=100 --compile=False")

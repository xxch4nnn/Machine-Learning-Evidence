import sys
import os
import time

def main():
    print("Starting Environment Verification...")
    print("-" * 40)

    # ---------------------------------------------------------
    # 1. PyTorch Verification
    # ---------------------------------------------------------
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            print(f"PyTorch CUDA   : AVAILABLE")
            print(f"Device Name    : {device_name}")
            print(f"Capability     : {capability}")
        else:
            print("CRITICAL FAILURE: torch.cuda.is_available() returned False.")
            sys.exit(1)
    except ImportError:
        print("CRITICAL FAILURE: PyTorch is not installed.")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL FAILURE: PyTorch check raised exception: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. OpenCV Verification
    # ---------------------------------------------------------
    try:
        import cv2
        # Check for CUDA support via getBuildInformation or getCudaEnabledDeviceCount
        # cv2.cuda.getCudaEnabledDeviceCount() is a reliable runtime check.
        cuda_devices = 0
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        except AttributeError:
            # cv2.cuda module not available
            pass
        
        if cuda_devices > 0:
            print(f"OpenCV         : CUDA-ENABLED ({cuda_devices} devices)")
        else:
            print("WARNING: OpenCV is CPU-only. Threading is mandatory.")
            
    except ImportError:
        print("CRITICAL FAILURE: OpenCV is not installed.")
        sys.exit(1)

    # ---------------------------------------------------------
    # 3. Tensor Test (5000x5000)
    # ---------------------------------------------------------
    print("-" * 40)
    print("Running Tensor Test (5000x5000 Matrix Mul)...")
    try:
        # Allocate on GPU
        a = torch.randn(5000, 5000, device='cuda')
        b = torch.randn(5000, 5000, device='cuda')
        
        # Warmup / Synchronize
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Compute
        c = torch.matmul(a, b)
        
        # Synchronize again to measure actual computation time
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"Tensor Test    : PASSED")
        print(f"Execution Time : {end_time - start_time:.4f} seconds")
        
    except Exception as e:
        print(f"CRITICAL FAILURE: Tensor Test failed. Error: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 4. Artifact Check
    # ---------------------------------------------------------
    print("-" * 40)
    
    # Path Resolution: Relative to script -> ../Data/PianoMotion10M/models/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_models_path = os.path.join(script_dir, "..", "Data", "PianoMotion10M", "models")
    
    # Fallback to project root if relative path fails (optional, but good practice)
    # However, for strict verification, we stick to the expected structure.
    models_dir = os.path.abspath(relative_models_path)
    
    print(f"Checking Artifacts in: {models_dir}")
    
    required_files = [
        "rf_model.pkl",
        "scaler.pkl",
        "selected_features.pkl"
    ]
    
    missing_files = []
    
    if not os.path.isdir(models_dir):
         print(f"CRITICAL FAILURE: Models directory does not exist.")
         sys.exit(1)
         
    for fname in required_files:
        fpath = os.path.join(models_dir, fname)
        if not os.path.exists(fpath):
            missing_files.append(fname)
            
    if missing_files:
        print(f"CRITICAL FAILURE: Missing critical artifacts: {missing_files}")
        sys.exit(1)
        
    print("Artifact Check : PASSED")
    print("-" * 40)
    print("VERIFICATION SUCCESSFUL: System is Ready.")

if __name__ == "__main__":
    main()

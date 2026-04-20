import platform
import sys
import torch
import subprocess

def get_detailed_report():
    print(f"{'='*20} 系统环境详细报告 {'='*20}")
    
    # 1. Windows 版本信息
    print(f"【操作系统】")
    print(f"- 系统名称: {platform.system()} {platform.release()}")
    print(f"- 详细版本: {platform.version()}")
    print(f"- 架构: {platform.machine()}")

    # 2. Python 版本信息
    print(f"\n【Python 环境】")
    print(f"- Python 版本: {sys.version.split()[0]}")
    print(f"- 执行路径: {sys.executable}")

    # 3. GPU 与 CUDA 信息 (PyTorch 视角)
    print(f"\n【GPU 硬件与 PyTorch 适配】")
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"- GPU 型号: {prop.name}")
        print(f"- 算力等级 (Compute Capability): {prop.major}.{prop.minor}")
        print(f"- 总显存: {prop.total_memory / 1024**2:.2f} MB")
        print(f"- PyTorch 编译版本: {torch.__version__}")
        print(f"- PyTorch 内部 CUDA 版本: {torch.version.cuda}")
    else:
        print("- CUDA 不可用！")

    # 4. NVIDIA 驱动与 系统 CUDA 版本 (命令行视角)
    print(f"\n【NVIDIA 驱动与系统工具】")
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
        for line in nvidia_smi.split('\n'):
            if "Driver Version" in line:
                print(f"- {line.strip()}")
                break
    except:
        print("- 无法获取 nvidia-smi 信息")

    print(f"{'='*56}")

if __name__ == "__main__":
    get_detailed_report()

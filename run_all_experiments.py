"""
一键运行所有对比实验
功能：依次训练五个模型并生成对比报告
日期：2026年
"""
import os
import sys
import subprocess
import json
import time
import gc
from config import Config

CONDA_PYTHON = r'C:\Users\18656\.conda\envs\chongju\python.exe'

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TIMM_VERBOSITY'] = '0'


def clear_gpu_memory():
    """清理GPU内存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            print("GPU内存已清理")
    except Exception as e:
        print(f"清理GPU内存时出错: {str(e)}")


def run_command(script_path, args, description):
    """运行命令并返回结果"""
    print(f"\n{'='*70}")
    print(f"开始: {description}")
    print(f"Python: {CONDA_PYTHON}")
    print(f"脚本: {script_path} {args}")
    print('='*70)
    
    env = os.environ.copy()
    env['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    try:
        result = subprocess.run(
            [CONDA_PYTHON, script_path] + args.split(),
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
            timeout=7200
        )
        print(f"\n✓ {description} 完成!")
        
        print("等待GPU内存释放...")
        time.sleep(10)
        clear_gpu_memory()
        time.sleep(5)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} 失败!")
        print(f"错误代码: {e.returncode}")
        clear_gpu_memory()
        return False
    except subprocess.TimeoutExpired:
        print(f"\n✗ {description} 超时!")
        clear_gpu_memory()
        return False
    except Exception as e:
        print(f"\n✗ {description} 发生异常: {str(e)}")
        clear_gpu_memory()
        return False


def main():
    print("="*80)
    print("柑橘虫害多视角检测对比实验")
    print("="*80)
    
    import torch
    print(f"\n使用设备: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch版本: {torch.__version__}")
    print()
    
    Config.create_directories()
    
    data_dir = './datasets'
    backbone = 'convnext_tiny'
    epochs = 30
    
    results = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    train_script = os.path.join(base_dir, 'train_comparison_models.py')
    main_script = os.path.join(base_dir, 'main.py')
    
    # 1. 训练单视角模型
    success = run_command(
        train_script,
        f'--data_dir {data_dir} --model_type single_view --backbone {backbone} --epochs {epochs}',
        '训练单视角模型'
    )
    if success:
        results['single_view'] = '已完成'
    
    # 2. 训练简单拼接模型
    success = run_command(
        train_script,
        f'--data_dir {data_dir} --model_type simple_concat --backbone {backbone} --epochs {epochs}',
        '训练简单拼接模型'
    )
    if success:
        results['simple_concat'] = '已完成'
    
    # 3. 训练CNN-LSTM模型
    success = run_command(
        train_script,
        f'--data_dir {data_dir} --model_type cnn_lstm --backbone {backbone} --epochs {epochs}',
        '训练CNN-LSTM模型'
    )
    if success:
        results['cnn_lstm'] = '已完成'
    
    # 4. 训练CNN-LSTM-Attention模型
    success = run_command(
        train_script,
        f'--data_dir {data_dir} --model_type cnn_lstm_attention --backbone {backbone} --epochs {epochs}',
        '训练CNN-LSTM-Attention模型'
    )
    if success:
        results['cnn_lstm_attention'] = '已完成'
    
    # 5. 训练注意力融合模型
    success = run_command(
        main_script,
        f'--data_dir {data_dir} --backbone {backbone} --epochs {epochs} --model_name attention_fusion',
        '训练注意力融合模型'
    )
    if success:
        results['attention_fusion'] = '已完成'
    
    # 保存实验结果
    results_path = os.path.join(Config.CHECKPOINT_PATH, 'experiment_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print("所有实验执行完毕!")
    print("实验结果:")
    for model, status in results.items():
        print(f"  - {model}: {status}")
    print("\n结果文件:")
    print(f"  - 单视角模型: checkpoints/single_view/")
    print(f"  - 简单拼接模型: checkpoints/simple_concat/")
    print(f"  - CNN-LSTM模型: checkpoints/cnn_lstm/")
    print(f"  - CNN-LSTM-Attention模型: checkpoints/cnn_lstm_attention/")
    print(f"  - 注意力融合模型: checkpoints/attention_fusion/")
    print(f"  - 实验记录: {results_path}")
    print("="*80)


if __name__ == '__main__':
    main()

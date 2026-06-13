import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# 动态获取当前脚本所在的绝对目录
SCRIPT_DIR = Path(__file__).resolve().parent
SEED_FILE = SCRIPT_DIR / "seed.smi"
TEMPLATE_TOML = SCRIPT_DIR / "sampling_template.toml"
RUN_TOML = SCRIPT_DIR / "running_mol2mol.toml"

def read_seed_smiles(seed_file_path):
    """从 seed.smi 文件中读取第一个 SMILES 作为参考分子"""
    with open(seed_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                return line.split()[0]
    raise ValueError(f"无法从 {seed_file_path} 中读取有效的 SMILES")

def prepare_toml_config(seed_smiles, num_samples=200):
    """读取模板并生成最终的 running_mol2mol.toml"""
    with open(TEMPLATE_TOML, 'r', encoding='utf-8') as f:
        config_content = f.read()

    # 动态替换占位符
    config_content = config_content.replace('PLACEHOLDER_SMILES', seed_smiles)
    config_content = config_content.replace('num_smiles = 200', f'num_smiles = {num_samples}')
    
    with open(RUN_TOML, 'w', encoding='utf-8') as f:
        f.write(config_content)
        
    return RUN_TOML

def main():
    print("\n" + "="*60)
    print("  ChemoMap Core - REINVENT 4 Library Generator (Mol2Mol)")
    print("="*60)
    print(f"[INFO] 当前工作目录识别为: {SCRIPT_DIR}")

    # 定义其他外部路径常量
    TARGET_DIR = "/home/xiayun-huang/ChemoMap/library"
    OUTPUT_CSV = "library.csv"

    # 1. 读取种子分子 SMILES
    try:
        seed_smiles = read_seed_smiles(SEED_FILE)
        print(f"[INFO] 成功读取种子分子 SMILES: {seed_smiles}")
    except Exception as e:
        print(f"[CRITICAL] 读取种子文件失败: {e}")
        sys.exit(1)

    # 2. 准备最终的 TOML 配置文件
    try:
        run_toml = prepare_toml_config(seed_smiles, num_samples=200)
        print(f"[INFO] 成功生成运行配置文件: {run_toml}")
    except Exception as e:
        print(f"[CRITICAL] 准备配置文件失败: {e}")
        sys.exit(1)

    # 3. 调用 REINVENT 4 进行分子生成
    print("[RUNNING] 正在调用 REINVENT 4 生成 200 个相似分子 (GPU加速中)...")
    cmd = ["reinvent", str(RUN_TOML.name)]
    
    try:
        # 切换到脚本所在目录运行，确保 library.csv 生成在脚本同级目录
        process = subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)
        print("[INFO] REINVENT 4 分子生成任务完成！")
    except subprocess.CalledProcessError as e:
        print(f"[CRITICAL] REINVENT 4 运行失败，退出码: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("[CRITICAL] 找不到 'reinvent' 命令，请检查 REINVENT 4 环境。")
        sys.exit(1)

    # 4. 导出并拷贝 library.csv 到目标目录
    generated_csv_path = SCRIPT_DIR / OUTPUT_CSV
    if generated_csv_path.exists():
        target_path = Path(TARGET_DIR) / OUTPUT_CSV
        try:
            shutil.copy(generated_csv_path, target_path)
            print(f"[SUCCESS] 成功将 {OUTPUT_CSV} 拷贝至: {target_path}")
        except Exception as e:
            print(f"[CRITICAL] 拷贝文件失败: {e}")
            sys.exit(1)
    else:
        print(f"[CRITICAL] REINVENT 4 运行完成但未找到输出文件 {OUTPUT_CSV}")
        sys.exit(1)

    # 5. 按时间重命名备份
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"library_{timestamp}.csv"
    backup_path = Path(TARGET_DIR) / backup_name
    
    try:
        shutil.copy(target_path, backup_path)
        print(f"[SUCCESS] 成功按时间戳备份为: {backup_path}")
    except Exception as e:
        print(f"[CRITICAL] 备份文件失败: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("  分子库生成与备份流水线圆满完成！")
    print("="*60)

if __name__ == "__main__":
    main()

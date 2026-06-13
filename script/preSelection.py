import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# ==========================================
# 模块 1: 战役时空阵地物理创建算子
# ==========================================
def create_campaign_directory(base_path="/home/xiayun-huang/ChemoMap/Campaign"):
    """
    以当前系统时间（精确到秒）为名，在 Campaign 目录下创建独立的战役文件夹
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_folder = Path(base_path) / timestamp
    
    try:
        campaign_folder.mkdir(parents=True, exist_ok=True)
        print(f"\n[INIT] 成功开辟本次筛选战役的独立物理阵地:")
        print(f"  -> {campaign_folder.resolve()}")
        return campaign_folder
    except Exception as e:
        print(f"[CRITICAL] 创建战役文件夹失败: {e}")
        sys.exit(1)

# ==========================================
# 模块 2: 靶点边界交互与 target.txt 固化算子
# ==========================================
def setup_targets(campaign_folder):
    """
    交互式收集主靶点与负向靶点，并规范化写入 target.txt
    """
    print("\n" + "-"*40)
    print("        配置筛选靶点边界 (Target Setup)")
    print("-"*40)
    
    # 1. 主靶点录入
    main_target = input("请输入主靶点 (Main Target) 的 PDB 编号 (例如 7eu8): ").strip().lower()
    while not main_target:
        main_target = input("主靶点不能为空，请重新输入: ").strip().lower()
        
    # 2. 负向靶点数量控制
    try:
        off_target_num = input("请输入需要配置的负向靶点 (Off Target) 数量 (直接回车默认为 0): ").strip()
        off_target_num = int(off_target_num) if off_target_num else 0
    except ValueError:
        print("[WARN] 输入非法，负向靶点数量自动重置为 0。")
        off_target_num = 0
        
    # 3. 循环收集负向靶点
    off_targets = []
    for i in range(off_target_num):
        off_t = input(f"  请输入第 {i+1} 个负向靶点的 PDB 编号: ").strip().lower()
        if off_t and off_t != main_target:
            off_targets.append(off_t)
            
    # 4. 数据持久化落地至 target.txt
    target_file_path = campaign_folder / "target.txt"
    try:
        with open(target_file_path, "w", encoding="utf-8") as f:
            f.write(f"MainTarget={main_target}\n")
            f.write(f"OffTarget={','.join(off_targets)}\n")
        print(f"[SUCCESS] 靶点清单已固化至: {target_file_path.name}")
    except Exception as e:
        print(f"[ERROR] 写入 target.txt 失败: {e}")

# ==========================================
# 模块 3: 种子分子交互与 seed.smi 固化算子
# ==========================================
def setup_seed_smiles(campaign_folder):
    """
    交互式收集小分子演化种子，规范化单向写入当前战役文件夹
    """
    print("\n" + "-"*40)
    print("       配置先导种子分子 (Seed SMILES Setup)")
    print("-"*40)
    
    seed_smiles = input("请输入种子分子的 SMILES 结构式 (例如 CC(=O)Nc1ccc(O)cc1): ").strip()
    while not seed_smiles:
        seed_smiles = input("种子分子 SMILES 不能为空，请重新输入: ").strip()
        
    seed_file_path = campaign_folder / "seed.smi"
    try:
        with open(seed_file_path, "w", encoding="utf-8") as f:
            f.write(f"{seed_smiles}\n")
        print(f"[SUCCESS] 种子构型已固化至当前战役: {seed_file_path.name}")
    except Exception as e:
        print(f"[ERROR] 写入 seed.smi 失败: {e}")

# ==========================================
# 模块 4: 执行核心（TOML & 脚本）自我克隆算子
# ==========================================
def clone_execution_core(campaign_folder, script_base="/home/xiayun-huang/ChemoMap/script"):
    """
    【全新升级】：反向从全局脚本区将执行核心克隆到新战役目录，实现环境全归档自包含
    """
    print("\n" + "-"*40)
    print("      部署战役自包含执行核心 (Core Cloning)")
    print("-"*40)
    
    src_dir = Path(script_base)
    files_to_clone = ["sampling_template.toml", "libGenerator.py"]
    
    for filename in files_to_clone:
        src_file = src_dir / filename
        dest_file = campaign_folder / filename
        
        if src_file.exists():
            try:
                shutil.copy(str(src_file), str(dest_file))
                print(f"[CLONE] 已成功将执行核心部署至战役阵地: {filename}")
            except Exception as e:
                print(f"[ERROR] 克隆 {filename} 失败: {e}")
        else:
            print(f"[CRITICAL] 核心源文件缺失，无法同步归档: {src_file.resolve()}")

# ==========================================
# 模块 5: 外部总控调度主接口
# ==========================================
def main():
    """
    预筛选战役初始化总控入口
    """
    print("\n" + "="*60)
    print("  ChemoMap Core - 筛选战役预初始化状态机 (自包含归档版)")
    print("="*60)
    
    # 1. 创建时间戳文件夹
    campaign_folder = create_campaign_directory()
    
    # 2. 生成靶点配置文件
    setup_targets(campaign_folder)
    
    # 3. 生成种子分子文件
    setup_seed_smiles(campaign_folder)
    
    # 4. 反向克隆核心依赖至当前战役目录
    clone_execution_core(campaign_folder)
    
    print("\n" + "="*60)
    print(f"  [SUCCESS] 战役自包含初始化完毕！执行核心与配置文件已全部就位。")
    print("="*60)

if __name__ == "__main__":
    main()

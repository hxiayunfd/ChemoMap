import csv
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from rdkit import Chem
except ImportError:
    print("[ERROR] 缺少 rdkit 库，请通过 'pip install rdkit' 安装。")
    sys.exit(1)

# 全局定义并行数量，你可以随时在这里修改（比如改成 4 或 8）
MAX_WORKERS = 2 

def parse_grid_file(grid_file_path):
    """解析口袋三维坐标参数配置文件"""
    grid_params = {}
    with open(grid_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip().lower()
            if key in ['center_x', 'center_y', 'center_z', 'size_x', 'size_y', 'size_z']:
                grid_params[key] = float(val.strip())
    return grid_params

def run_gnina_single(args):
    """调用 GNINA 对单个 SDF 分子进行对接 (适配多进程的参数解包)"""
    protein_path, ligand_sdf_path, grid_params, output_sdf_path, exhaustiveness = args
    
    cmd = [
        "gnina", "-r", str(protein_path), "-l", str(ligand_sdf_path), "-o", str(output_sdf_path),
        "--center_x", str(grid_params['center_x']), "--center_y", str(grid_params['center_y']),
        "--center_z", str(grid_params['center_z']), "--size_x", str(grid_params['size_x']),
        "--size_y", str(grid_params['size_y']), "--size_z", str(grid_params['size_z'])
    ]
    if exhaustiveness is not None:
        cmd.extend(["--exhaustiveness", str(exhaustiveness)])
    
    try:
        # 依然保持静默运行，避免多个进程同时打印导致终端乱码
        process = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ligand_sdf_path.name, process.returncode == 0
    except Exception:
        return ligand_sdf_path.name, False

def post_process_results(output_folder, summary_csv_path):
    """从所有 _docked.sdf 中提取最优结合 Pose 并导出打分报告 CSV"""
    results_summary = []
    for docked_sdf in sorted(Path(output_folder).glob("*_docked.sdf")):
        compound_name = docked_sdf.stem.replace("_docked", "")
        try:
            suppl = Chem.SDMolSupplier(str(docked_sdf), removeHs=False)
            mol = next(suppl)
            if mol is None: continue
            
            affinity = None
            for prop_name in ["CNNscore", "CNNaffinity", "min_affinity_score"]:
                if mol.HasProp(prop_name):
                    try: affinity = float(mol.GetProp(prop_name)); break
                    except: pass
            
            smiles = ""
            try: smiles = Chem.MolToSmiles(mol)
            except: pass
                
            if affinity is not None:
                results_summary.append({"Compound": compound_name, "SMILES": smiles, "Affinity Score": affinity})
        except Exception:
            pass

    with open(summary_csv_path, mode="w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=["Compound", "SMILES", "Affinity Score"])
        writer.writeheader()
        writer.writerows(results_summary)

def main(protein, ligand_dir, grid_file, summary_csv, exhaustiveness=None):
    """并行对接主流水线"""
    print("\n" + "="*60)
    print(f"  ChemoMap Core - GNINA Engine (Parallel x{MAX_WORKERS})")
    print("="*60)
    
    protein_path = Path(protein)
    ligand_folder = Path(ligand_dir)
    
    # 清理旧的对接结果
    for old_file in ligand_folder.glob("*_docked.sdf"):
        try: old_file.unlink()
        except: pass

    # 读取口袋坐标参数
    try:
        grid_params = parse_grid_file(Path(grid_file))
    except Exception as e:
        print(f"[CRITICAL] 口袋配置文件读取失败: {e}")
        return

    # 查找有效配体文件
    ligand_files = [f for f in ligand_folder.glob("*.sdf") 
                    if "_docked" not in f.name and "temp" not in f.name and f.stat().st_size > 200]

    if not ligand_files:
        print("[CRITICAL] 未在目标目录下检测到任何可以进行对接的健全配体结构！")
        return

    print(f"[INFO] 成功扫描到 {len(ligand_files)} 个有效配体，开始 {MAX_WORKERS} 线程并行对接...")

    # 准备多进程所需的参数包
    tasks = []
    for ligand_file in ligand_files:
        output_sdf_path = ligand_folder / f"{ligand_file.stem}_docked.sdf"
        tasks.append((protein_path, ligand_file, grid_params, output_sdf_path, exhaustiveness))

    # 使用进程池并行执行对接
    success_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_ligand = {executor.submit(run_gnina_single, task): task[1].name for task in tasks}
        
        # 监听任务完成情况
        for i, future in enumerate(as_completed(future_to_ligand)):
            ligand_name, success = future.result()
            if success:
                success_count += 1
                print(f"[DOCKING] ({i+1}/{len(ligand_files)}) 成功: {ligand_name}")
            else:
                print(f"[WARN]    ({i+1}/{len(ligand_files)}) 失败: {ligand_name}")

    # 解析结果并归档
    if success_count > 0:
        post_process_results(ligand_folder, summary_csv)
        print(f"\n[SUCCESS] GNINA 并行对接完成！成功 {success_count} 个，结果已汇总。")
    else:
        print("\n[CRITICAL] GNINA 底层异常退出，无有效对接结果。")

if __name__ == "__main__":
    pass

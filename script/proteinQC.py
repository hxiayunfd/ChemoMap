import os
import sys
import csv
from pathlib import Path

SCRIPT_DIR = Path("/home/xiayun-huang/ChemoMap/script")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

try:
    import ligandPrepare as lp
    import gnina as gn
except ImportError as e:
    print(f"[CRITICAL] 无法加载上游脚本模块，请检查 {SCRIPT_DIR} 下的文件是否存在。")
    print(f"错误信息: {e}")
    sys.exit(1)


def main():
    print("="*60)
    print("  ChemoMap 蛋白质量控制流水线 (Protein QC Pipeline)")
    print("="*60)

    pdb_id = input("请输入待质检的 PDB 编号 (例如 7eu8): ").strip().lower()
    if not pdb_id:
        print("[ERROR] PDB 编号不能为空！")
        return

    BASE_DIR = Path("/home/xiayun-huang/ChemoMap")
    # 受体蛋白路径（请确保该路径下的文件存在且格式正确）
    protein_path = BASE_DIR / f"protein/meeko_pdbqt/{pdb_id}.pdbqt"
    tool_csv = BASE_DIR / f"library/toolCompound/toolCompound_{pdb_id}.csv"
    grid_file = BASE_DIR / f"protein/grid/{pdb_id}.txt"
    summary_csv = BASE_DIR / f"protein_qc_{pdb_id}_results.csv"

    if not protein_path.exists():
        print(f"[ERROR] 未找到受体蛋白文件: {protein_path}")
        return
    if not tool_csv.exists():
        print(f"[ERROR] 未找到对应的阳性对照小分子 CSV 文件: {tool_csv}")
        return
    if not grid_file.exists():
        print(f"[ERROR] 未找到该 PDB 的口袋配置文件: {grid_file}\n        请先运行上游脚本扫描并锁定该靶点的结合口袋。")
        return

    print(f"\n[STEP 1] 路径解析与依赖检查成功:")
    print(f"  -> 受体蛋白: {protein_path}")
    print(f"  -> 工具化合物清单: {tool_csv}")
    print(f"  -> 选定的口袋坐标: {grid_file}")

    tool_output_dir = BASE_DIR / f"library/toolCompound/toolCompound_{pdb_id}"
    tool_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[STEP 2] 已确认工具化合物输出目录:\n  -> {tool_output_dir}")

    # [STEP 3] 调用 ligandPrepare 模块进行 3D SDF 构象转化与加氢
    print(f"\n[STEP 3] 正在调用 ligandPrepare 模块进行 3D SDF 构象转化与加氢...")
    success_count = 0
    fail_count = 0
    
    try:
        with open(tool_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            print(f"  [DEBUG] 读取到的 CSV 表头为: {reader.fieldnames}")
            
            for row in reader:
                # 完美兼容 Compound/Name/name 以及 SMILES/Smiles/smiles 等各种表头写法
                name = row.get('Compound', row.get('Name', row.get('name', ''))).strip()
                smiles = row.get('SMILES', row.get('Smiles', row.get('smiles', ''))).strip()
                
                if name and smiles:
                    output_sdf = tool_output_dir / f"{name}.sdf"
                    print(f"  [PREPARE] 正在处理: {name}")
                    # 调用函数并判断返回值
                    if lp.process_to_3d_sdf(name, smiles, output_sdf):
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    print(f"  [WARN] 跳过一行数据，未找到有效的 Compound/Name 或 SMILES: {row}")
                    fail_count += 1
                    
    except Exception as e:
        print(f"[CRITICAL] 调用 ligandPrepare 失败: {e}")
        return

    # 核心拦截：如果没有任何一个配体生成成功，直接终止程序，不再继续跑对接
    if success_count == 0:
        print(f"\n[CRITICAL] 配体准备阶段全军覆没！成功: {success_count}, 失败: {fail_count}")
        print("请检查上方 [PREPARE] 的报错信息。")
        return
    else:
        print(f"\n[INFO] 配体准备完成！成功生成 {success_count} 个 SDF 文件，失败 {fail_count} 个。")

    # [STEP 4] 调用 gnina 模块启动 GPU 质检对接
    print(f"\n[STEP 4] 正在调用 gnina 模块启动 GPU 质检对接 (Exhaustiveness: 32)...")
    try:
        gn.main(
            protein=str(protein_path),
            ligand_dir=str(tool_output_dir),
            grid_file=str(grid_file),
            summary_csv=str(summary_csv),
            exhaustiveness=32
        )
    except Exception as e:
        print(f"[CRITICAL] 调用 gnina 对接失败: {e}")
        return

    print("\n" + "="*60)
    print(f"[SUCCESS] 蛋白 {pdb_id.upper()} 质检流水线运行完毕！")
    print(f"[INFO] 阳性分子对接 Pose 已存入: {tool_output_dir}")
    print(f"[INFO] 质检打分报告已生成至: {summary_csv.resolve()}")
    print("="*60)

if __name__ == "__main__":
    main()

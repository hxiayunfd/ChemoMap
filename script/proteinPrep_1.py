# -*- coding: utf-8 -*-
import os
import requests
import numpy as np
import subprocess
import tempfile

# ==================== 配置路径 ====================
BASE_DIR = "/home/xiayun-huang/ChemoMap"
RAW_PDB_DIR = os.path.join(BASE_DIR, "protein/rawPDB")
PREP_PDBQT_DIR = os.path.join(BASE_DIR, "protein/meeko_pdbqt")
GRID_DIR = os.path.join(BASE_DIR, "protein/grid")

# 自动创建所有工作文件夹
os.makedirs(RAW_PDB_DIR, exist_ok=True)
os.makedirs(PREP_PDBQT_DIR, exist_ok=True)
os.makedirs(GRID_DIR, exist_ok=True)

# 常见的结晶辅助剂、离子和溶剂黑名单
# 注意：千万不要把你真正的药物配体（比如 JC9）写在这里！
LIGAND_BLACKLIST = {
    "HOH", "DOD", "WAT", "SOL",  
    "SO4", "PO4", "NO3", "CO3", "EDT", "ACT", "FOR", "AET", 
    "CL", "NA", "MG", "ZN", "EDT", "CA", "K", "NI", "CU", "MN", "FE", 
    "GOL", "EDO", "PEG", "PG4", "PGE", "PEO", "DTT", "DMS", "BME", 
    "TMS", "NAG", "MAN", "BMA", "FUC", "GAL" 
}

def download_pdb(pdb_id):
    """从 RCSB PDB 下载原始 PDB 文件"""
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = os.path.join(RAW_PDB_DIR, f"{pdb_id}.pdb")
    
    print(f" 正在从 PDB 官网下载 {pdb_id.upper()}...")
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f" 成功下载并保存至: {output_path}")
            return output_path
        else:
            print(f" 错误: 无法下载 PDB 编号 {pdb_id.upper()}，状态码: {response.status_code}")
            return None
    except Exception as e:
        print(f" 下载过程中出现异常: {e}")
        return None

def auto_calculate_all_grids(raw_pdb_path, pdb_id, padding=12.0):
    """
    【核心功能保留】
    扫描 PDB 内所有异质小分子，过滤黑名单后，为每一个独立的小分子计算一套完整的 Grid 坐标
    """
    pdb_id = pdb_id.lower()
    all_ligands = {}
    
    print(f" ⚙️ 正在全量扫描 {pdb_id.upper()} 结构中的异质小分子...")
    
    with open(raw_pdb_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
                # 过滤黑名单
                if res_name in LIGAND_BLACKLIST:
                    continue
                
                chain_id = line[21:22].strip()
                if not chain_id: 
                    chain_id = "A" 
                res_seq = line[22:26].strip()
                
                ligand_key = f"{chain_id}_{res_name}_{res_seq}"
                
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    if ligand_key not in all_ligands:
                        all_ligands[ligand_key] = []
                    all_ligands[ligand_key].append([x, y, z])
                except ValueError:
                    continue

    grid_files_created = []

    # 如果清洗后没有发现有效小分子，转为全蛋白盲对接口袋
    if not all_ligands:
        print(" ⚠️ 提示：清除结晶辅剂后未发现内源性配体小分子，将自动探测全蛋白中心作为全局口袋...")
        protein_coords = []
        with open(raw_pdb_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("ATOM"):
                    try:
                        protein_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    except ValueError: continue
        if protein_coords:
            coords = np.array(protein_coords)
            center = np.mean(coords, axis=0)
            grid_txt_path = os.path.join(GRID_DIR, f"{pdb_id}_global.txt")
            with open(grid_txt_path, 'w', encoding='utf-8') as out_f:
                out_f.write(f"# Global Blind Docking Box for {pdb_id.upper()}\n")
                out_f.write(f"center_x = {center[0]:.3f}\ncenter_y = {center[1]:.3f}\ncenter_z = {center[2]:.3f}\n")
                out_f.write("size_x = 30.000\nsize_y = 30.000\nsize_z = 30.000\n")
            grid_files_created.append(grid_txt_path)
            print(f" -> 盲对接 Grid 文件已生成: {grid_txt_path}")
        return grid_files_created

    # 遍历所有被识别出的小分子，依次提取 Grid Box 
    print(f"\n==================== 🎯 识别到以下共结晶结合口袋 ({len(all_ligands)}个) ====================")
    
    for idx, (lig_key, points) in enumerate(all_ligands.items(), start=1):
        coords = np.array(points)
        center = np.mean(coords, axis=0) 
        
        # 计算跨度尺寸并加上 padding
        min_bounds = np.min(coords, axis=0)
        max_bounds = np.max(coords, axis=0)
        ligand_sizes = max_bounds - min_bounds
        
        size_x = ligand_sizes[0] + padding
        size_y = ligand_sizes[1] + padding
        size_z = ligand_sizes[2] + padding
        
        # 写入独立的 txt 文件
        grid_txt_path = os.path.join(GRID_DIR, f"{pdb_id}_{lig_key}.txt")
        with open(grid_txt_path, 'w', encoding='utf-8') as out_f:
            out_f.write(f"# Auto-generated Grid Box for Ligand: {lig_key}\n")
            out_f.write(f"center_x = {center[0]:.3f}\n")
            out_f.write(f"center_y = {center[1]:.3f}\n")
            out_f.write(f"center_z = {center[2]:.3f}\n")
            out_f.write(f"size_x = {size_x:.3f}\n")
            out_f.write(f"size_y = {size_y:.3f}\n")
            out_f.write(f"size_z = {size_z:.3f}\n")
            
        grid_files_created.append(grid_txt_path)
        print(f" 口袋 [{idx}] -> 标识符: {lig_key} (共 {len(points)} 个重原子)")
        print(f"    位置: 中心 ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) | 尺寸 ({size_x:.1f}, {size_y:.1f}, {size_z:.1f})")
        print(f"    配置文件已保存至: {grid_txt_path}\n")
        
    print("============================================================================")
    return grid_files_created

def fix_and_prepare_protein(input_path, pdb_id):
    """使用 MGLTools 官方脚本进行最稳健的受体准备"""
    pdb_id = pdb_id.lower()
    final_pdbqt_path = os.path.join(PREP_PDBQT_DIR, f"{pdb_id}.pdbqt")
    
    print(f" 正在使用 MGLTools 对 {pdb_id.upper()} 进行专业级受体准备...")
    try:
        # 1. 使用 PDBFixer 进行基础的修复（补全重原子、加氢等）
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        
        fixer = PDBFixer(filename=input_path)
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        
        # 将修复后的结构保存到临时 PDB 文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_f:
            temp_pdb_path = tmp_f.name
            PDBFile.writeFile(fixer.topology, fixer.positions, tmp_f, keepIds=True)

        # 2. 调用 MGLTools 的官方脚本 prepare_receptor4.py 生成 PDBQT
        # 这个脚本是 AutoDock 官方提供的，兼容性最好，绝对不会出现 RDKit 的价态报错
        cmd = [
            "prepare_receptor4.py",
            "-r", temp_pdb_path,
            "-o", final_pdbqt_path,
            "-A", "checkhydrogens", # 检查并保留氢原子
            "-U", "nphs_lps_waters" # 合并非极性氢，移除水分子（标准受体准备流程）
        ]
        
        print(f" 正在调用 AutoDock 官方工具生成 PDBQT...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 清理临时文件
        os.remove(temp_pdb_path)
        
        if result.returncode != 0:
            print(f"❌ MGLTools 转换失败: {result.stderr}")
            return None
            
        print(f" 受体 PDBQT 转换成功: {final_pdbqt_path}")
        return final_pdbqt_path
        
    except Exception as e:
        print(f"❌ 蛋白修复或转换失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    print("="*60)
    print("    ChemoMap 工作流 - 受体准备 + 多配体口袋全自动扫描")
    print("="*60)
    
    pdb_input = input("请输入对接靶点的 PDB 编号 (例如 '7eu8'): ").strip()
    
    if len(pdb_input) != 4:
        print("❌ 错误：PDB 编号通常由 4 个字符组成。")
    else:
        # 1. 下载
        raw_file = download_pdb(pdb_input)
        
        if raw_file:
            # 2. 扫描所有独立小分子配体并生成各自的 Grid 文件
            grid_files = auto_calculate_all_grids(raw_file, pdb_input, padding=12.0)
            
            # 3. PDBFixer 核心修复清理并转 PDBQT
            success_path = fix_and_prepare_protein(raw_file, pdb_input)
            
            if success_path and grid_files:
                print(f"\n🎉 运行成功！已自动输出 {len(grid_files)} 个独立口袋的 Grid 配置文件。")
                print(f"受体 PDBQT 已就绪: {success_path}")
            else:
                print("\n❌ 运行过程中出现部分失败，请检查上方报错日志。")

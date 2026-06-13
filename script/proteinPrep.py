#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import requests
import subprocess
import numpy as np

BASE_DIR = "/home/xiayun-huang/ChemoMap"
RAW_PDB_DIR = os.path.join(BASE_DIR, "protein/rawPDB")
PREP_PDBQT_DIR = os.path.join(BASE_DIR, "protein/meeko_pdbqt")
GRID_DIR = os.path.join(BASE_DIR, "protein/grid")

os.makedirs(RAW_PDB_DIR, exist_ok=True)
os.makedirs(PREP_PDBQT_DIR, exist_ok=True)
os.makedirs(GRID_DIR, exist_ok=True)

LIGAND_BLACKLIST = {
    "HOH", "DOD", "WAT", "SOL",
    "SO4", "PO4", "NO3", "CO3", "EDT", "ACT", "FOR", "AET",
    "CL", "NA", "MG", "ZN", "EDT", "CA", "K", "NI", "CU", "MN", "FE",
    "GOL", "EDO", "PEG", "PG4", "PGE", "PEO", "DTT", "DMS", "BME",
    "TMS", "NAG", "MAN", "BMA", "FUC", "GAL"
}

def download_pdb(pdb_id):
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
    pdb_id = pdb_id.lower()
    all_ligands = {}
    
    print(f" ⚙️ 正在全量扫描 {pdb_id.upper()} 结构中的异质小分子...")
    
    with open(raw_pdb_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
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

    grid_candidate_list = []

    if not all_ligands:
        print(" ⚠️ 提示：清除结晶辅剂后未发现内源性配体小分子，自动探测全蛋白中心...")
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
            grid_candidate_list.append({
                "name": "Global_Blind_Docking",
                "center": center,
                "size": [30.0, 30.0, 30.0],
                "info": "全蛋白几何中心"
            })
        return grid_candidate_list

    for lig_key, points in all_ligands.items():
        coords = np.array(points)
        center = np.mean(coords, axis=0)
        
        min_bounds = np.min(coords, axis=0)
        max_bounds = np.max(coords, axis=0)
        ligand_sizes = max_bounds - min_bounds
        
        size_x = ligand_sizes[0] + padding
        size_y = ligand_sizes[1] + padding
        size_z = ligand_sizes[2] + padding
        
        grid_candidate_list.append({
            "name": lig_key,
            "center": center,
            "size": [size_x, size_y, size_z],
            "info": f"共结晶配体 (含 {len(points)} 个重原子)"
        })
        
    return grid_candidate_list

def export_final_grid(grid_data, pdb_id):
    pdb_id = pdb_id.lower()
    final_txt_path = os.path.join(GRID_DIR, f"{pdb_id}.txt")
    
    try:
        with open(final_txt_path, 'w', encoding='utf-8') as out_f:
            out_f.write(f"# Auto-generated Grid Box for Target PDB: {pdb_id.upper()}\n")
            out_f.write(f"# Based on selected source: {grid_data['name']}\n")
            out_f.write(f"center_x = {grid_data['center'][0]:.3f}\n")
            out_f.write(f"center_y = {grid_data['center'][1]:.3f}\n")
            out_f.write(f"center_z = {grid_data['center'][2]:.3f}\n")
            out_f.write(f"size_x = {grid_data['size'][0]:.3f}\n")
            out_f.write(f"size_y = {grid_data['size'][1]:.3f}\n")
            out_f.write(f"size_z = {grid_data['size'][2]:.3f}\n")
        print(f" Success! 选定坐标已成功导出至唯一配置文件: {final_txt_path}")
        return True
    except Exception as e:
        print(f"❌ 导出 Grid 文件失败: {e}")
        return False

def fix_and_prepare_protein(input_path, pdb_id):
    pdb_id = pdb_id.lower()
    fixed_pdb_path = os.path.join(RAW_PDB_DIR, f"{pdb_id}_fixed.pdb")
    final_pdbqt_path = os.path.join(PREP_PDBQT_DIR, f"{pdb_id}.pdbqt")
    
    print(f" 正在启动 PDBFixer 对 {pdb_id.upper()} 进行全量加氢修复...")
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        
        fixer = PDBFixer(filename=input_path)
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
       # fixer.addMissingHydrogens(7.0)
        
        temp_all_h_pdb = fixed_pdb_path + ".tmp"
        with open(temp_all_h_pdb, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
            
        print(f" 🛡️ 正在执行地毯式非极性氢强力裁剪（彻底摧毁 32768 段错误）...")
        clean_lines = []
        with open(temp_all_h_pdb, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    
                    # 只要原子名包含 H，就开始进行极性/非极性硬核鉴别
                    if "H" in atom_name:
                        # 强行排除法：只放行标准的极性氢
                        # 1. 主链及部分侧链核心氮上的极性氢
                        if atom_name in ["H", "H1", "H2", "H3"]:
                            clean_lines.append(line)
                            continue
                        # 2. 特定极性残基侧链N/O/S上的极性氢
                        if res_name == "ARG" and any(x in atom_name for x in ["HE", "HH11", "HH12", "HH21", "HH22"]):
                            clean_lines.append(line)
                            continue
                        if res_name == "ASN" and any(x in atom_name for x in ["HD21", "HD22"]):
                            clean_lines.append(line)
                            continue
                        if res_name == "GLN" and any(x in atom_name for x in ["HE21", "HE22"]):
                            clean_lines.append(line)
                            continue
                        if res_name == "HIS" and any(x in atom_name for x in ["HD1", "HE2"]):
                            clean_lines.append(line)
                            continue
                        if res_name == "TRP" and "HE1" in atom_name:
                            clean_lines.append(line)
                            continue
                        if res_name == "LYS" and any(x in atom_name for x in ["HZ1", "HZ2", "HZ3"]):
                            clean_lines.append(line)
                            continue
                        if res_name in ["SER", "THR", "CYS"] and "HG" in atom_name:
                            clean_lines.append(line)
                            continue
                        if res_name == "TYR" and "HH" in atom_name:
                            clean_lines.append(line)
                            continue
                            
                        # 其余所有挂在碳上的脂肪族/芳香族氢 (如 1HD1, HA, HB) 一律干掉
                        continue
                        
                clean_lines.append(line)
                
        with open(fixed_pdb_path, 'w') as f:
            f.writelines(clean_lines)
            
        if os.path.exists(temp_all_h_pdb):
            os.remove(temp_all_h_pdb)
            
        success = convert_pdb_to_pdbqt(fixed_pdb_path, final_pdbqt_path)
        return final_pdbqt_path if success else None
    except Exception as e:
        print(f"❌ PDB 修复转换失败: {e}")
        return None

def convert_pdb_to_pdbqt(pdb_path, pdbqt_path):
    cmd_name = "/home/xiayun-huang/miniconda3/bin/prepare_receptor4"
    
    if not os.path.exists(cmd_name):
        python_bin_dir = os.path.dirname(sys.executable)
        cmd_name = os.path.join(python_bin_dir, "prepare_receptor4")
        if not os.path.exists(cmd_name):
            cmd_name = "prepare_receptor4"
            if not shutil.which(cmd_name):
                print(f" ❌ 错误: 无法在系统中定位 'prepare_receptor4' 可执行文件。")
                return False
        
    cmd = [cmd_name, "-r", pdb_path, "-o", pdbqt_path, "-A", "checkhydrogens", "-U", "nvals"]
    
    print(f" 🚀 执行转换命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if os.path.exists(pdbqt_path):
            cleaned_lines = []
            with open(pdbqt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        atom_type = line[77:80].strip()
                        if atom_type == "O":
                            line = line[:77] + "OA\n"
                        elif atom_type == "N":
                            line = line[:77] + "NA\n"
                    cleaned_lines.append(line)
            with open(pdbqt_path, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)
            print(f" 🛡️ [后处理] 已强制纠正漏网的旧原子类型 (O->OA, N->NA)")
            
        print(f" ✅ [ADT_py3] 受体标准 PDBQT 格式化完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f" ❌ ADT 转换失败！")
        print(f" 🔴 错误详情:\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("    ChemoMap 工作流 - 受体准备 + 多配体口袋交互式选择")
    print("="*60)
    
    pdb_input = input("请输入对接靶点的 PDB 编号 (例如 '7eu8'): ").strip()
    
    if len(pdb_input) != 4:
        print("❌ 错误：PDB 编号通常由 4 个字符组成。")
    else:
        raw_file = download_pdb(pdb_input)
        
        if raw_file:
            candidates = auto_calculate_all_grids(raw_file, pdb_input, padding=12.0)
            
            selected_grid = None
            if not candidates:
                print("❌ 未能在当前 PDB 结构中识别到任何有效的蛋白质原子或口袋。")
            else:
                print(f"\n🎯 共检测到 {len(candidates)} 个候选对接结合口袋，请选择：")
                print("-" * 60)
                for i, cand in enumerate(candidates, start=1):
                    c_str = f"({cand['center'][0]:.2f}, {cand['center'][1]:.2f}, {cand['center'][2]:.2f})"
                    s_str = f"({cand['size'][0]:.1f}, {cand['size'][1]:.1f}, {cand['size'][2]:.1f})"
                    print(f" [{i}] 标识符: {cand['name']}")
                    print(f"     类别: {cand['info']}")
                    print(f"     坐标: 中心 {c_str} | 扩展尺寸 {s_str}\n")
                print("-" * 60)
                
                while True:
                    try:
                        user_choice = input(f"请输入你想使用的口袋序号 (1-{len(candidates)}): ").strip()
                        choice_idx = int(user_choice) - 1
                        if 0 <= choice_idx < len(candidates):
                            selected_grid = candidates[choice_idx]
                            print(f" 已锁定口袋: {selected_grid['name']}")
                            break
                        else:
                            print(f"❌ 输入错误，请输入 1 到 {len(candidates)} 之间的数字。")
                    except ValueError:
                        print("❌ 输入无效，请输入正确的整数序号。")
            
            grid_success = False
            if selected_grid:
                grid_success = export_final_grid(selected_grid, pdb_input)
            
            success_path = fix_and_prepare_protein(raw_file, pdb_input)
            
            if success_path and grid_success:
                print(f"\n🎉 流水线运行成功！")
                print(f"   -> 受体 PDBQT 已就绪: {success_path}")
                print(f"   -> 口袋配置文件已覆盖更新为: {os.path.join(GRID_DIR, f'{pdb_input.lower()}.txt')}")
            else:
                print("\n⚠️ 流水线部分模块运行失败，请检查上方日志。")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import glob
import shutil

# ================== ⚙️ Configuration ==================
GRID_TXT_DIR = "/home/xiayun-huang/ChemoMap/protein/grid"
PDBQT_DIR = "/home/xiayun-huang/ChemoMap/protein/meeko_pdbqt"
OUTPUT_DIR = "/home/xiayun-huang/ChemoMap/protein/grid"
AUTOGRID_CMD = "autogrid4"
SPACING = 0.375

LIGAND_TYPES = ["A", "C", "HD", "N", "NA", "OA", "SA", "F", "CL", "BR", "I"]

def parse_txt_file(txt_path):
    """Extract center and size parameters from the txt file"""
    params = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key.startswith('center_') or key.startswith('size_'):
                    try:
                        params[key] = float(value)
                    except ValueError:
                        pass
    return params

def generate_gpf(receptor_pdbqt, txt_params, output_gpf):
    npts_x = int((txt_params['size_x'] / SPACING + 0.5) // 2 * 2)
    npts_y = int((txt_params['size_y'] / SPACING + 0.5) // 2 * 2)
    npts_z = int((txt_params['size_z'] / SPACING + 0.5) // 2 * 2)
    
    receptor_filename = os.path.basename(receptor_pdbqt)
    base_stem = receptor_filename[:-6]

    with open(output_gpf, 'w') as f:
        # 💡 核心修改：强制写入绝对路径，不给 autogrid4 任何读取同名残留缓存的机会！
        f.write(f"receptor {os.path.abspath(receptor_pdbqt)}\n")
        
        f.write(f"gridcenter {txt_params['center_x']:.3f} {txt_params['center_y']:.3f} {txt_params['center_z']:.3f}\n")
        f.write(f"spacing {SPACING:.3f}\n")
        
        ligand_types_str = " ".join(LIGAND_TYPES)
        f.write(f"ligand_types {ligand_types_str}\n")
        
        f.write(f"npts {npts_x} {npts_y} {npts_z}\n")
        
        for atom_type in LIGAND_TYPES:
            f.write(f"map {base_stem}.{atom_type}.map\n")
            
        f.write(f"elecmap {base_stem}.e.map\n")
        f.write(f"dsolvmap {base_stem}.d.map\n")
        f.write("dielectric -0.1465\n")

def run_autogrid(gpf_file, log_file, work_dir):
    """Call autogrid4 command"""
    cmd = [AUTOGRID_CMD, "-p", gpf_file, "-l", log_file]
    print(f"🚀 Executing: {' '.join(cmd)}")
    try:
        # 使用 subprocess 运行，并捕获错误输出
        result = subprocess.run(cmd, cwd=work_dir, check=True, capture_output=True, text=True)
        print(f"✅ AutoGrid completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ AutoGrid failed!")
        print(f"🔴 Error details:\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}")

def main():
    print("🔍 Starting batch AutoGrid generation...")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    txt_files = glob.glob(os.path.join(GRID_TXT_DIR, "*.txt"))
    
    if not txt_files:
        print(f"❌ No .txt files found in {GRID_TXT_DIR}!")
        return

    for txt_file in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        print(f"\n{'='*40}")
        print(f"📂 Processing target: {base_name}")
        
        pdbqt_file = os.path.join(PDBQT_DIR, f"{base_name}.pdbqt")
        
        if not os.path.exists(pdbqt_file):
            print(f"❌ Error: {base_name}.pdbqt not found in {PDBQT_DIR}, skipped.")
            continue
            
        # 拷贝受体文件到工作目录
        dest_pdbqt = os.path.join(OUTPUT_DIR, os.path.basename(pdbqt_file))
        if not os.path.exists(dest_pdbqt):
            shutil.copy(pdbqt_file, dest_pdbqt)
            print(f"📋 Receptor copied to work dir: {os.path.basename(pdbqt_file)}")

        # 1. 解析 txt 参数
        params = parse_txt_file(txt_file)
        if len(params) < 6:
            print(f"❌ Error: Insufficient parameters in {txt_file}, skipped.")
            continue
            
        # 2. 生成标准的 .gpf 文件
        gpf_file = os.path.join(OUTPUT_DIR, f"{base_name}.gpf")
        generate_gpf(pdbqt_file, params, gpf_file)
        print(f"📝 Generated GPF file: {gpf_file}")
        
        # 3. 运行 autogrid4
        log_file = f"{base_name}.glg"
        run_autogrid(os.path.basename(gpf_file), log_file, OUTPUT_DIR)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
REINVENT4 本地自动化分子生成与质控一体化脚本 (精简输出 + 严格递增命名版)
"""

import os
import re
import pandas as pd
import subprocess
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import FilterCatalog

# ================= 1. 配置区域与自动轮次识别 =================
BASE_DIR = "/home/xiayun-huang/KDproject"
INPUT_CSV = os.path.join(BASE_DIR, "cns_selected_hits.csv")          
TEMPLATE_TOML = os.path.join(BASE_DIR, "sampling_template.toml")      

# 自动解析输入文件名中的轮次数字，并自动加 1
# 比如输入包含 r3，则当前新库轮次自动判定为 4
input_filename = os.path.basename(INPUT_CSV)
match = re.search(r'r(\d+)', input_filename)
ROUND_NUM = int(match.group(1)) + 1 if match else 4  # 如果没匹配到，默认作为第 4 轮

OUTPUT_CSV = os.path.join(BASE_DIR, f"Enamine_r{ROUND_NUM}.csv")

# 循环中转的临时文件
TEMP_SMI = os.path.join(BASE_DIR, "temp_mol2mol.smi")
TEMP_TOML = os.path.join(BASE_DIR, "temp_config.toml")
TEMP_OUTPUT = os.path.join(BASE_DIR, "temp_output.csv")                

# 精确本地模型路径
PRESET_MODEL = "/home/xiayun-huang/REINVENT4-main/prior/mol2mol_medium_similarity.prior"
REINVENT_CMD = "reinvent"  

# 采样规模配置
MOLS_PER_COMPOUND = 200   

# ================= 2. 质控过滤器定义 =================
def is_cns_compliant(mol):
    """判断化合物是否符合经典中枢神经系统(CNS)药物的理化性质准入标准"""
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        
        if (mw <= 450) and (1.0 <= logp <= 4.5) and (tpsa <= 90) and (hbd <= 2) and (hba <= 7):
            return True
    except:
        return False
    return False

# ================= 3. 准备工作 =================
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"未找到本地输入骨架文件: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV, header=0)
if 'compound' not in df.columns or 'smiles' not in df.columns:
    df.columns = ['compound', 'smiles'] + list(df.columns[2:])

num_compounds = len(df)
print(f"📊 识别到输入文件: {input_filename} -> 当前新库判定为第 {ROUND_NUM} 轮")
print(f"📊 共有 {num_compounds} 个输入骨架，每个生成 {MOLS_PER_COMPOUND} 个衍生物。")

# 清理历史残余
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

# 初始化 PAINS 过滤器
fparams = FilterCatalog.FilterCatalogParams()
fparams.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
pains_catalog = FilterCatalog.FilterCatalog(fparams)

# 全局化合物编号计数器（从 1 开始递增，对应最终 CSV 文件的真实物理行号）
global_mol_idx = 1

# ================= 4. 循环调用 REINVENT4 生成并原位清洗 =================
for index, row in df.iterrows():
    name = row['compound']
    smiles = row['smiles']
    print(f"[{index+1}/{num_compounds}] 正在处理骨架: {name}")

    # 1. 生成临时的 .smi 骨架文件
    with open(TEMP_SMI, 'w') as f:
        f.write(f"{smiles}\n")
    
    # 2. 读取模板并替换占位符
    with open(TEMPLATE_TOML, 'r') as f:
        toml_content = f.read()
    
    toml_content = toml_content.replace("[[MODEL_PATH]]", PRESET_MODEL)
    toml_content = toml_content.replace("[[SMILES_FILE]]", TEMP_SMI)
    toml_content = toml_content.replace("[[NUM_SMILES]]", str(MOLS_PER_COMPOUND))
    
    with open(TEMP_TOML, 'w') as f:
        f.write(toml_content)
    
    # 3. 运行 REINVENT4
    cmd = [REINVENT_CMD, TEMP_TOML]
    try:
        subprocess.run(cmd, check=True)
        
        # 4. 原位质控过滤、重新规范化命名，并直接追加写入两列最终文件
        if os.path.exists(TEMP_OUTPUT):
            gen_df = pd.read_csv(TEMP_OUTPUT)
            
            # 兼容处理 REINVENT4 的输出 SMILES 列名
            smiles_col = 'SMILES' if 'SMILES' in gen_df.columns else ('smiles' if 'smiles' in gen_df.columns else gen_df.columns[0])
            
            valid_mols = []
            for _, gen_row in gen_df.iterrows():
                smi = gen_row[smiles_col]
                if pd.isna(smi):
                    continue
                
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    continue
                
                # 严格过滤网
                if not is_cns_compliant(mol) or pains_catalog.HasMatch(mol):
                    continue
                
                # 质控通过：自动生成递增命名的标准行字典
                valid_mols.append({
                    'Compound': f"Enamine_r{ROUND_NUM}_{global_mol_idx}",
                    'SMILES': Chem.MolToSmiles(mol) # 统一规范化 SMILES 输出
                })
                global_mol_idx += 1
            
            # 如果有幸存分子，直接追加写入到极简 CSV 里
            if valid_mols:
                chunk_df = pd.DataFrame(valid_mols)
                chunk_df.to_csv(OUTPUT_CSV, mode='a', index=False, header=not os.path.exists(OUTPUT_CSV), encoding='utf-8')
                
            os.remove(TEMP_OUTPUT)
        else:
            print(f"⚠️ 警告：REINVENT4 未产生临时输出文件 {TEMP_OUTPUT}")

    except subprocess.CalledProcessError:
        print(f"❌ 骨架 [{name}] 的 REINVENT4 运行失败。")

# ================= 5. 清理临时中转 =================
for temp_file in [TEMP_SMI, TEMP_TOML]:
    if os.path.exists(temp_file):
        os.remove(temp_file)

if os.path.exists(OUTPUT_CSV):
    print(f"\n🎉 任务全部圆满完成！最终两列标准合规库已生成至: {OUTPUT_CSV}")
    print(f"📊 最终总共收录了 {global_mol_idx - 1} 个中枢渗透性质完美的衍生物。")
else:
    print("\n❌ 质控卡得太死或未成功生成分子，未产生有效最终文件。")

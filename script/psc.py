# -*- coding: utf-8 -*-
"""
虚拟筛选数据分析终极脚本 (双受体比值 + CNS过滤 + PAINS剔除 + Tanimoto 0.7 骨架聚类完美版)
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import FilterCatalog
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem import rdFingerprintGenerator

def is_cns_compliant(mol):
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

def cluster_molecules_butina(mol_list, similarity_threshold=0.7):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [morgan_gen.GetFingerprint(mol) for mol in mol_list]
    num_mols = len(fps)
    
    dist_matrix = []
    for i in range(1, num_mols):
        sims = BulkTanimotoSimilarity(fps[i], fps[:i])
        for sim in sims:
            dist_matrix.append(1.0 - sim)
            
    dist_threshold = 1.0 - similarity_threshold
    
    from rdkit.ML.Cluster import Butina
    clusters = Butina.ClusterData(dist_matrix, num_mols, dist_threshold, isDistData=True, reordering=True)
    
    return clusters

def run_ultimate_screening_pipeline(
    csv_path="/home/xiayun-huang/KDproject/Enamine_r4_docking_results.csv", 
    output_csv="/home/xiayun-huang/KDproject/Enamine_r4_selected_hits.csv"
):
    print("==================================================")
    print(f"正在读取原始虚拟筛选数据: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到输入文件，请确认路径: {csv_path}")
        
    df = pd.read_csv(csv_path)
    orig_cols = df.columns.tolist()
    
    if len(orig_cols) < 5:
        raise ValueError("输入CSV文件的列数不足5列，请确保包含：化合物名称、SMILES、NMDA、Opioid、Selectivity。")
        
    col_mapping = {
        orig_cols[0]: 'compound',
        orig_cols[1]: 'smiles',
        orig_cols[2]: 'NMDA_Affinity',
        orig_cols[3]: 'Opiod_Affinity',
        orig_cols[4]: 'Original_Selectivity'
    }
    df = df.rename(columns=col_mapping)
    
    # 1. 【数据清洗】
    df_clean = df.dropna(subset=['NMDA_Affinity', 'Opiod_Affinity', 'Original_Selectivity']).copy()
    df_clean['NMDA_Affinity'] = pd.to_numeric(df_clean['NMDA_Affinity'], errors='coerce')
    df_clean['Opiod_Affinity'] = pd.to_numeric(df_clean['Opiod_Affinity'], errors='coerce')
    df_clean['Original_Selectivity'] = pd.to_numeric(df_clean['Original_Selectivity'], errors='coerce')
    df_clean = df_clean.dropna(subset=['NMDA_Affinity', 'Opiod_Affinity', 'Original_Selectivity']).copy()
    
    # 2. 计算群体参数
    nmda_mean = df_clean['NMDA_Affinity'].mean()
    nmda_sd = df_clean['NMDA_Affinity'].std()
    sel_mean = df_clean['Original_Selectivity'].mean()
    sel_sd = df_clean['Original_Selectivity'].std()
    
    # 3. 统计学动态筛选条件
    # 正常双负数体系下，比值越大越好
    cond_sel_high = df_clean['Original_Selectivity'] > (sel_mean + sel_sd)
    cond_sel_ultra = df_clean['Original_Selectivity'] > (sel_mean + 3 * sel_sd)
    
    cond_nmda_high = df_clean['NMDA_Affinity'] < (nmda_mean - nmda_sd)
    cond_nmda_ultra = df_clean['NMDA_Affinity'] < (nmda_mean - 3 * nmda_sd)
    
    # 【细节优化】：增加了最低活性门槛限制（NMDA < -4.0），防止结合力极弱的假阳性漏网
    cond_special_good = (df_clean['NMDA_Affinity'] < -4.0) & (df_clean['Opiod_Affinity'] > 0)
    
    # 彻底跑偏的分子（NMDA打分为正，完全不结合）直接一票否决
    cond_exclude = df_clean['NMDA_Affinity'] > 0
    
    # 多通道联动筛选
    df_clean['Class_1'] = cond_nmda_high & (cond_sel_high | cond_special_good) & (~cond_exclude)
    df_clean['Class_2'] = cond_nmda_ultra & (~cond_exclude)
    df_clean['Class_3'] = (cond_sel_ultra | cond_special_good) & (~cond_exclude)
    
    df_hit = df_clean[df_clean['Class_1'] | df_clean['Class_2'] | df_clean['Class_3']].copy()
    
    if df_hit.empty:
        print("[提示] 未筛选出任何符合对接打分条件的化合物分子。")
        return

    # 4. 【深度过滤器】CNS药效团属性控制 + PAINS假阳性片段剔除
    fparams = FilterCatalog.FilterCatalogParams()
    fparams.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog.FilterCatalog(fparams)
    
    passed_rows = []
    passed_mols = []
    
    for idx, row in df_hit.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            continue
        if not is_cns_compliant(mol):
            continue
        if pains_catalog.HasMatch(mol):
            continue
            
        passed_rows.append(row)
        passed_mols.append(mol)
        
    if not passed_rows:
        print("❌ 所有初选分子均在 CNS/PAINS 质控中被淘汰。")
        return
        
    df_audited = pd.DataFrame(passed_rows).reset_index(drop=True)
    print(f"-> 通过 CNS性质与 PAINS 质控审计的分子数: {len(df_audited)}")

    # 5. 【多样性化学空间聚类】基于 Tanimoto > 0.7 
    print("\n====== 开始进行分子 Morgan 指纹拓扑聚类 (Tanimoto > 0.7) ======")
    
    clusters = cluster_molecules_butina(passed_mols, similarity_threshold=0.7)
    print(f"-> 结构化学空间共划分为 {len(clusters)} 个独立的 Cluster 骨架簇。")
    
    selected_indices = []
    for cluster_id, cluster_members in enumerate(clusters):
        member_list = list(cluster_members)
        best_member_idx = min(member_list, key=lambda idx: df_audited.loc[idx, 'NMDA_Affinity'])
        selected_indices.append(best_member_idx)
        
    df_final_hits = df_audited.loc[selected_indices].copy()
    
    # 6. 【定制化极简导出】
    df_output = df_final_hits[['compound', 'smiles']]
    df_output.to_csv(output_csv, index=False, encoding='utf-8')
    
    print("\n==================================================")
    print("🎉 全流程精细化药物化学虚拟筛选圆满完成！")
    print(f"-> 原始有效池: {len(df_clean)} 个分子")
    print(f"-> 精选出的分子数: {len(df_output)}")
    print("==================================================")

if __name__ == "__main__":
    run_ultimate_screening_pipeline()

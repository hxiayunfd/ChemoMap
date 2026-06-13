import pandas as pd
import os
import requests
from chembl_webresource_client.new_client import new_client

# 配置路径
BASE_DIR = "/home/xiayun-huang/ChemoMap"
OUTPUT_DIR = os.path.join(BASE_DIR, "library", "toolCompound")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_uniprot_from_pdb(pdb_id):
    """【关键修复】从 PDB 动态抓取真正的 UniProt ID"""
    url = "https://data.rcsb.org/graphql"
    query = '{"query": "{ entry(entry_id: \\"%s\\") { polymer_entities { rcsb_polymer_entity_container_identifiers { uniprot_ids } } } }"}' % pdb_id.upper()
    try:
        res = requests.post(url, data=query, headers={'Content-Type': 'application/json'}).json()
        entities = res.get("data", {}).get("entry", {}).get("polymer_entities", [])
        for e in entities:
            uids = e.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids", [])
            # 过滤掉常见的非靶点蛋白（如溶菌酶 P00720）
            valid_uids = [u for u in uids if u != "P00720"]
            if valid_uids: return valid_uids[0]
    except: return None
    return None

def fetch_compounds(pdb_id, limit=50):
    # 1. 动态获取 UniProt
    uni_id = get_uniprot_from_pdb(pdb_id)
    if not uni_id:
        print(f" ❌ 无法从 {pdb_id} 解析到合法受体 UniProt，请检查 PDB ID。")
        return

    print(f" 🧬 PDB: {pdb_id} -> 锁定 UniProt: {uni_id}")
    
    # 2. 锁定靶点
    targets = new_client.target.filter(target_components__accession=uni_id)
    if not targets:
        print(" ❌ ChEMBL 中未检索到该靶点。")
        return
    
    target_id = targets[0]['target_chembl_id']
    print(f" 🎯 发现靶点: {targets[0]['pref_name']} (ID: {target_id})")

    # 3. 拉取去重数据
    res = new_client.activity.filter(
        target_chembl_id=target_id,
        standard_type__in=['IC50', 'Ki'],
        standard_value__lt=50,
        standard_units='nM'
    ).only(['molecule_chembl_id', 'canonical_smiles', 'molecule_pref_name'])

    compounds = {}
    for item in res:
        m_id = item.get('molecule_chembl_id')
        if m_id and m_id not in compounds:
            compounds[m_id] = {
                'Compound': m_id,
                'SMILES': item.get('canonical_smiles')
            }
        if len(compounds) >= limit: break
            
    # 4. 保存
    df = pd.DataFrame(list(compounds.values()))
    path = os.path.join(OUTPUT_DIR, f"toolCompound_{pdb_id.lower()}.csv")
    df.to_csv(path, index=False)
    print(f" ✅ 成功为 {pdb_id} 生成 {len(df)} 个唯一分子，保存至: {path}")

if __name__ == "__main__":
    pdb = input("请输入 PDB Code: ").strip()
    fetch_compounds(pdb)

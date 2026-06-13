#!/usr/bin/env python3
"""
模块: 工具化合物生成器
Module: Tool Compound Generator — ChEMBL Positive Controls

从 ChEMBL 数据库获取靶点的已知活性化合物，用于蛋白质量控制和对接验证。

流程:
1. PDB ID → RCSB GraphQL API → UniProt ID
2. UniProt ID → ChEMBL Web Resource API → 活性化合物
3. 筛选高活性化合物 (IC50/Ki < 50 nM)
4. 导出 CSV 文件

工具依赖:
    - requests: conda env "requests"
    - chembl_webresource_client: conda env "chembl"

用法:
    from tool_compound_generator import ToolCompoundGenerator

    gen = ToolCompoundGenerator()
    csv_path = gen.fetch_compounds("4dkl")
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DIRS

logger = logging.getLogger(__name__)


class ToolCompoundGenerator:
    """
    工具化合物生成器

    从 ChEMBL 获取靶点的已知活性化合物。
    所有网络操作通过 conda run 在对应环境中执行。
    """

    def __init__(self, config: Optional[dict] = None, conda_manager=None):
        """
        Args:
            config: 配置字典
            conda_manager: CondaManager 实例
        """
        self.config = config or {}

        if conda_manager is None:
            from conda_manager import get_conda_manager
            conda_manager = get_conda_manager()
        self.cm = conda_manager

    def _run_python(self, tool: str, script: str, timeout: int = 60) -> str:
        """在指定工具的 conda 环境中运行 Python 脚本"""
        result = self.cm.run_python(tool, script, timeout=timeout)
        return result.stdout

    # ============================================================
    # UniProt ID 解析
    # ============================================================

    def get_uniprot_from_pdb(self, pdb_id: str) -> Optional[str]:
        """
        通过 RCSB GraphQL API 从 PDB ID 解析 UniProt ID

        Args:
            pdb_id: 4 字符 PDB 编号

        Returns:
            UniProt ID，失败返回 None
        """
        pdb_id = pdb_id.upper()

        script = f"""
import json, requests
pdb_id = {pdb_id!r}

query = {{
    "query": f'{{{{ entry(entry_id: "{pdb_id}") {{{{ polymer_entities {{{{ rcsb_polymer_entity_container_identifiers {{{{ uniprot_ids }}}} }} }} }} }} }}'
}}

try:
    r = requests.post(
        "https://data.rcsb.org/graphql",
        json=query,
        headers={{"Content-Type": "application/json"}},
    ).json()
    entities = r.get("data", {{}}).get("entry", {{}}).get("polymer_entities", [])
    for e in entities:
        uids = e.get("rcsb_polymer_entity_container_identifiers", {{}}).get("uniprot_ids", [])
        valid = [u for u in uids if u != "P00720"]  # 排除 T4 溶菌酶
        if valid:
            print(json.dumps({{"uniprot_id": valid[0]}}))
            break
    else:
        print(json.dumps({{"uniprot_id": None}}))
except Exception as e:
    print(json.dumps({{"uniprot_id": None, "error": str(e)}}))
"""
        try:
            output = self._run_python("requests", script)
            result = json.loads(output.strip())
            uni = result.get("uniprot_id")
            if uni:
                logger.info(f"PDB {pdb_id} → UniProt {uni}")
            return uni
        except Exception as e:
            logger.error(f"UniProt resolution failed: {e}")
            return None

    # ============================================================
    # ChEMBL 化合物获取
    # ============================================================

    def fetch_compounds(
        self,
        pdb_id: str,
        limit: int = 50,
        activity_threshold: float = 50.0,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        从 ChEMBL 获取靶点的已知活性化合物

        Args:
            pdb_id: PDB 编号
            limit: 最大化合物数
            activity_threshold: 活性阈值 (nM)，默认 50 nM
            output_dir: 输出目录

        Returns:
            CSV 文件路径，失败返回 None
        """
        pdb_id_lower = pdb_id.lower()
        output_dir = Path(output_dir or DIRS["tool_compounds"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"toolCompound_{pdb_id_lower}.csv"

        # 1. 获取 UniProt ID
        uni_id = self.get_uniprot_from_pdb(pdb_id)
        if not uni_id:
            logger.error(f"Cannot resolve UniProt ID for {pdb_id}")
            return None

        # 2. 查询 ChEMBL
        script = f"""
import json, sys, csv

try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_OK = True
except ImportError:
    CHEMBL_OK = False

uni_id = {uni_id!r}
limit = {limit}
threshold = {activity_threshold}
output_path = {str(output_path)!r}

if not CHEMBL_OK:
    print(json.dumps({{"error": "chembl_webresource_client not installed", "compounds": []}}))
    sys.exit(0)

try:
    targets = new_client.target.filter(target_components__accession=uni_id)
    if not targets:
        print(json.dumps({{"error": f"No target for {{uni_id}}", "compounds": []}}))
        sys.exit(0)

    tid = targets[0]['target_chembl_id']
    tname = targets[0].get('pref_name', 'Unknown')

    activities = new_client.activity.filter(
        target_chembl_id=tid,
        standard_type__in=['IC50', 'Ki'],
        standard_value__lt=threshold,
        standard_units='nM',
    ).only(['molecule_chembl_id', 'canonical_smiles', 'molecule_pref_name'])

    compounds = {{}}
    for item in activities:
        mid = item.get('molecule_chembl_id')
        if mid and mid not in compounds:
            compounds[mid] = {{
                'Compound': mid,
                'SMILES': item.get('canonical_smiles'),
                'Name': item.get('molecule_pref_name', ''),
            }}
        if len(compounds) >= limit:
            break

    print(json.dumps({{
        "target_id": tid,
        "target_name": tname,
        "num_compounds": len(compounds),
        "compounds": list(compounds.values()),
    }}))
except Exception as e:
    print(json.dumps({{"error": str(e), "compounds": []}}))
"""
        try:
            output = self._run_python("chembl", script, timeout=120)
            result = json.loads(output.strip())

            if result.get("error"):
                logger.error(f"ChEMBL error: {result['error']}")
                return None

            compounds = result.get("compounds", [])
            if not compounds:
                logger.warning(f"No active compounds found for {pdb_id}")
                return None

            tname = result.get("target_name", "Unknown")
            logger.info(f"Target: {tname} ({result.get('target_id')})")

            # 3. 保存 CSV
            df = pd.DataFrame(compounds)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} tool compounds → {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"ChEMBL fetch failed: {e}")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gen = ToolCompoundGenerator()
    pdb_id = input("Enter PDB ID: ").strip()
    if pdb_id:
        result = gen.fetch_compounds(pdb_id)
        if result:
            print(f"✅ Compounds saved to: {result}")
        else:
            print("❌ Failed to fetch compounds")

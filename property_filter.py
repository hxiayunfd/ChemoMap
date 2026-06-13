#!/usr/bin/env python3
"""
模块: 理化性质筛选器
Module: Physicochemical Property Filter

使用 RDKit 计算化合物的理化性质，应用 Lipinski 五规则及扩展规则进行筛选。

筛选流程:
1. 计算分子量、LogP、HBD、HBA、TPSA、可旋转键、环数等
2. Lipinski 五规则检查（violations <= 1 通过）
3. PAINS 假阳性片段剔除
4. 用户自定义范围过滤

工具依赖:
    - RDKit: conda env "rdkit"

用法:
    from property_filter import PropertyFilter

    pf = PropertyFilter()
    props = pf.calculate_properties(["CCO", "CCN"])
    filtered = pf.apply_filters(props)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import PROPERTY_FILTERS, DIRS

logger = logging.getLogger(__name__)


class PropertyFilter:
    """
    理化性质筛选器

    计算 + 筛选一体。
    RDKit 操作通过 conda run 在指定环境中执行。
    RDKit 不可用时自动降级为模拟数据。
    """

    def __init__(self, config: Optional[Dict] = None, conda_manager=None):
        """
        Args:
            config: 筛选配置覆盖
            conda_manager: CondaManager 实例
        """
        self.config = {**PROPERTY_FILTERS, **(config or {})}

        if conda_manager is None:
            from conda_manager import get_conda_manager
            conda_manager = get_conda_manager()
        self.cm = conda_manager

    def _run_python(self, tool: str, script: str, timeout: int = 300) -> str:
        """在指定工具的 conda 环境中运行 Python 脚本"""
        result = self.cm.run_python(tool, script, timeout=timeout)
        return result.stdout

    # ============================================================
    # 性质计算
    # ============================================================

    def calculate_properties(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        批量计算理化性质（大量 SMILES 通过临时文件传递，避免命令行过长）
        """
        if not smiles_list:
            return pd.DataFrame()

        import tempfile
        task_file = Path(tempfile.mktemp(
            suffix=".json", prefix="props_",
            dir=Path.home() / ".cache",
        ))
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(json.dumps(smiles_list))
        remove_pains = str(self.config.get("remove_pains", True))

        script = f"""
import json, os
task_file = {str(task_file)!r}
with open(task_file) as f:
    smiles_list = json.load(f)
remove_pains = {remove_pains}

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, FilterCatalog
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False

# PAINS 过滤器
pains_catalog = None
if RDKIT_OK and remove_pains:
    try:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        pains_catalog = FilterCatalog(params)
    except: pass

records = []
for i, smi in enumerate(smiles_list):
    rec = {{"SMILES": smi, "index": i, "valid": False}}
    if not RDKIT_OK:
        rec["_mock"] = True
        records.append(rec)
        continue

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        records.append(rec)
        continue

    rec["valid"] = True
    rec["molecular_weight"] = Descriptors.MolWt(mol)
    rec["logp"] = Descriptors.MolLogP(mol)
    rec["hbd"] = Lipinski.NumHDonors(mol)
    rec["hba"] = Lipinski.NumHAcceptors(mol)
    rec["rotatable_bonds"] = Lipinski.NumRotatableBonds(mol)
    rec["tpsa"] = Descriptors.TPSA(mol)
    rec["ring_count"] = Lipinski.RingCount(mol)
    rec["aromatic_rings"] = Lipinski.NumAromaticRings(mol)
    rec["heavy_atom_count"] = mol.GetNumHeavyAtoms()
    rec["formal_charge"] = Chem.GetFormalCharge(mol)
    rec["fsp3"] = Descriptors.FractionCSP3(mol)

    rec["pains_alert"] = bool(pains_catalog and pains_catalog.HasMatch(mol))

    # Lipinski violations
    v = 0
    if rec["molecular_weight"] > 500: v += 1
    if rec["logp"] > 5: v += 1
    if rec["hbd"] > 5: v += 1
    if rec["hba"] > 10: v += 1
    rec["lipinski_violations"] = v
    rec["lipinski_pass"] = v <= 1

    records.append(rec)

print(json.dumps(records))
os.remove(task_file)
"""
        try:
            output = self._run_python("rdkit", script)
            records = json.loads(output.strip())
            df = pd.DataFrame(records)
            logger.info(f"Calculated properties for {len(df)} compounds")
            return df
        except Exception as e:
            logger.warning(f"RDKit calculation failed: {e}. Using mock data.")
            return self._mock_calculate(smiles_list)

    def _mock_calculate(self, smiles_list: List[str]) -> pd.DataFrame:
        """降级：模拟性质计算"""
        import random
        records = []
        for i, smi in enumerate(smiles_list):
            records.append({
                "SMILES": smi, "index": i, "valid": True,
                "molecular_weight": random.uniform(200, 600),
                "logp": random.uniform(-1, 6),
                "hbd": random.randint(0, 6),
                "hba": random.randint(0, 12),
                "rotatable_bonds": random.randint(0, 12),
                "tpsa": random.uniform(20, 160),
                "ring_count": random.randint(1, 7),
                "aromatic_rings": random.randint(0, 4),
                "heavy_atom_count": random.randint(10, 40),
                "formal_charge": 0,
                "fsp3": random.uniform(0.3, 0.7),
                "pains_alert": random.random() < 0.05,
                "lipinski_violations": random.randint(0, 3),
                "lipinski_pass": random.random() > 0.3,
                "_mock": True,
            })
        return pd.DataFrame(records)

    # ============================================================
    # 筛选
    # ============================================================

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用理化性质筛选规则（含 CNS 严格过滤）"""
        if df.empty:
            return df

        initial = len(df)
        removed = []

        # 1. 有效性
        if "valid" in df.columns:
            before = len(df)
            df = df[df["valid"] == True].copy()
            if len(df) < before:
                removed.append(f"invalid({before - len(df)})")

        # 2. 分子量 & LogP 范围
        for col, label in [("molecular_weight", "MW"), ("logp", "LogP")]:
            cfg = self.config.get(col)
            if cfg and col in df.columns:
                before = len(df)
                df = df[(df[col] >= cfg["min"]) & (df[col] <= cfg["max"])]
                if len(df) < before:
                    removed.append(f"{label}({before - len(df)})")

        # 3. 阈值过滤
        for col, label in [
            ("hbd", "HBD"), ("hba", "HBA"),
            ("rotatable_bonds", "RotB"), ("ring_count", "Rings"),
        ]:
            cfg = self.config.get(col)
            if cfg and col in df.columns:
                before = len(df)
                df = df[df[col] <= cfg["max"]]
                if len(df) < before:
                    removed.append(f"{label}({before - len(df)})")

        # 4. TPSA
        tpsa_cfg = self.config.get("tpsa")
        if tpsa_cfg and "tpsa" in df.columns:
            before = len(df)
            df = df[(df["tpsa"] >= tpsa_cfg["min"]) & (df["tpsa"] <= tpsa_cfg["max"])]
            if len(df) < before:
                removed.append(f"TPSA({before - len(df)})")

        # 5. Fsp3 (CNS 严格)
        fsp3_cfg = self.config.get("fsp3")
        if fsp3_cfg and "fsp3" in df.columns:
            before = len(df)
            df = df[df["fsp3"] > fsp3_cfg.get("min", 0.45)]
            if len(df) < before:
                removed.append(f"Fsp3(>{fsp3_cfg['min']}:{before - len(df)})")

        # 6. 强酸基团排除 (SMARTS)
        if self.config.get("remove_strong_acids", True) and "SMILES" in df.columns:
            smi_list = df["SMILES"].tolist()
            try:
                passed = self._check_no_strong_acid(smi_list)
                before = len(df)
                df = df.iloc[[i for i, ok in enumerate(passed) if ok]].reset_index(drop=True)
                if len(df) < before:
                    removed.append(f"StrongAcids({before - len(df)})")
            except Exception as e:
                logger.warning(f"Strong acid check failed: {e}")

        # 7. PAINS
        if self.config.get("remove_pains") and "pains_alert" in df.columns:
            before = len(df)
            df = df[df["pains_alert"] == False]
            if len(df) < before:
                removed.append(f"PAINS({before - len(df)})")

        # 8. Lipinski
        if self.config.get("require_lipinski", True) and "lipinski_pass" in df.columns:
            before = len(df)
            df = df[df["lipinski_pass"] == True]
            if len(df) < before:
                removed.append(f"Lipinski({before - len(df)})")

        logger.info(
            f"Filter: {initial} → {len(df)} "
            f"({' | '.join(removed) if removed else 'no removals'})"
        )
        return df.reset_index(drop=True)

    def _check_no_strong_acid(self, smiles_list: list) -> list:
        """SMARTS 检测强酸基团（羧酸/磺酸/四氮唑/磷酸），返回 bool 列表"""
        script = f"""
import json
from rdkit import Chem

smiles_list = json.loads({json.dumps(smiles_list)!r})
ACIDIC = [
    "[CX3](=O)[OX2H1]",
    "[SX4](=O)(=O)[OX2H1]",
    "[SX4](=O)[OX2H1]",
    "c1[nH]nnn1",
    "[PX4](=O)([OX2H1])[OX2H1]",
]
pats = [Chem.MolFromSmarts(p) for p in ACIDIC]
results = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        results.append(True)
        continue
    ok = not any(mol.HasSubstructMatch(p) for p in pats)
    results.append(ok)
print(json.dumps({{"passed": results}}))
"""
        output = self._run_python("rdkit", script)
        import json
        return json.loads(output.strip()).get("passed", [True] * len(smiles_list))

    # ============================================================
    # 便捷方法
    # ============================================================

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        smiles_column: str = "SMILES",
    ) -> pd.DataFrame:
        """
        一站式：从 SMILES DataFrame 计算性质并筛选

        Args:
            df: 输入 DataFrame（必须包含 SMILES 列）
            smiles_column: SMILES 列名

        Returns:
            筛选后的 DataFrame
        """
        if df.empty:
            return df

        smiles_list = df[smiles_column].tolist()
        prop_df = self.calculate_properties(smiles_list)

        # 保留原始 df 的其他列
        for col in df.columns:
            if col != smiles_column and col not in prop_df.columns:
                prop_df[col] = df[col].values[:len(prop_df)]

        return self.apply_filters(prop_df)

    def filter_and_save(
        self,
        input_df: pd.DataFrame,
        output_path: Path,
        smiles_column: str = "SMILES",
    ) -> pd.DataFrame:
        """
        完整筛选并保存 CSV

        Args:
            input_df: 输入 DataFrame
            output_path: 输出 CSV 路径
            smiles_column: SMILES 列名

        Returns:
            筛选后的 DataFrame
        """
        filtered = self.filter_dataframe(input_df, smiles_column=smiles_column)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(output_path, index=False)
        logger.info(f"Saved {len(filtered)} compounds to {output_path}")

        return filtered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pf = PropertyFilter()
    test_smiles = [
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "COC1=CC2=C(C=C1)C(=O)C(=CO2)C3=CC=C(C=C3)O",
        "invalid_xxx",
    ]
    test_df = pd.DataFrame({"SMILES": test_smiles, "score": [0.8, 0.6, 0.1]})
    result = pf.filter_and_save(
        test_df,
        DIRS["temp"] / "filtered_test.csv",
    )
    print(f"Input: {len(test_df)}, Output: {len(result)}")
    if not result.empty:
        cols = ["SMILES", "molecular_weight", "logp", "lipinski_pass"]
        available = [c for c in cols if c in result.columns]
        print(result[available].head())

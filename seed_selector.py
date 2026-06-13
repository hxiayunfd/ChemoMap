#!/usr/bin/env python3
"""
模块: 种子选择器
Module: Seed Selector — Docking Result Analysis & Seed Selection

从对接结果中挑选下一轮 Reinvent4 生成的种子分子。

挑选策略:
1. "top_n"      — 对接分数最好的 N 个
2. "threshold"  — 结合能低于阈值的全部化合物
3. "cluster"    — Butina Tanimoto 聚类后每簇选最优
4. "cns_cluster" — CNS 过滤 + PAINS 剔除 + 聚类挑优

增强过滤（可选）:
- CNS 药物性质过滤（血脑屏障穿透性）
- PAINS 假阳性片段剔除
- Butina Tanimoto 聚类（相似度 0.7）
- 双受体选择性分析

工具依赖:
    - RDKit: conda env "rdkit"

用法:
    from seed_selector import SeedSelector

    selector = SeedSelector()
    seeds = selector.select_seeds(docking_df, round_num=1)
    selector.export_seeds_to_smi(seeds, Path("seeds.smi"))
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import SEED_SELECTION, DIRS

logger = logging.getLogger(__name__)


class SeedSelector:
    """
    种子选择器

    多种策略挑选种子 + 增强过滤（CNS/PAINS/聚类/双受体分析）。
    RDKit 操作通过 conda run 执行。
    """

    def __init__(self, config: Optional[Dict] = None, conda_manager=None):
        """
        Args:
            config: 筛选配置覆盖
            conda_manager: CondaManager 实例
        """
        self.config = {**SEED_SELECTION, **(config or {})}

        if conda_manager is None:
            from conda_manager import get_conda_manager
            conda_manager = get_conda_manager()
        self.cm = conda_manager

    def _run_python(self, tool: str, script: str, timeout: int = 300) -> str:
        """在指定工具的 conda 环境中运行 Python 脚本（自动检测正确环境）"""
        output = self.cm.run_python(tool, script, timeout=timeout)
        return output.stdout

    # ============================================================
    # 严格 CNS 理化性质过滤 (MW<360, cLogP 2~4, TPSA<70, HBD≤1, HBA≤7, RB≤5, Fsp3>0.45, 中性/弱碱)
    # ============================================================

    def _cns_strict_properties_script(self, smiles_list):
        """RDKit 批量计算 CNS 严格物性（模块级函数引用，返回符合条件索引）"""
        import json as _json
        smi_json = _json.dumps(smiles_list)

        script = f"""
import json
smiles_list = json.loads({smi_json!r})
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen

# 酸性基团 SMARTS（羧酸、磺酸、四氮唑等）
ACIDIC_SMARTS = [
    "[CX3](=O)[OX2H1]",        # 羧酸
    "[SX4](=O)(=O)[OX2H1]",    # 磺酸
    "[SX4](=O)[OX2H1]",        # 亚磺酸
    "c1[nH]nnn1",              # 四氮唑
    "[PX4](=O)([OX2H1])[OX2H1]", # 磷酸
]

results = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        results.append(False)
        continue

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    fsp3 = Descriptors.FractionCSP3(mol)

    # 物性检查
    if not (mw < 360): results.append(False); continue
    if not (2.0 <= logp <= 4.0): results.append(False); continue
    if not (tpsa < 70): results.append(False); continue
    if not (hbd <= 1): results.append(False); continue
    if not (hba <= 7): results.append(False); continue
    if not (rb <= 5): results.append(False); continue
    if not (fsp3 > 0.45): results.append(False); continue

    # pKa 近似：排除强酸基团
    has_acidic = False
    for pat in ACIDIC_SMARTS:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(pat)):
            has_acidic = True
            break
    if has_acidic: results.append(False); continue

    results.append(True)

print(json.dumps({{"passed": results}}))
"""
        return script

    def apply_cns_strict_filter(self, df: pd.DataFrame, smiles_column: str = "SMILES") -> pd.DataFrame:
        """
        CNS 严格物性过滤:
          MW < 360, cLogP 2~4, TPSA < 70, HBD ≤ 1, HBA ≤ 7,
          RotBonds ≤ 5, Fsp3 > 0.45, 无强酸基团 (pKa 近似)
        """
        smiles_list = df[smiles_column].tolist() if smiles_column in df.columns else []
        if not smiles_list:
            return df

        script = self._cns_strict_properties_script(smiles_list)
        try:
            output = self._run_python("rdkit", script)
            import json
            passed = json.loads(output.strip()).get("passed", [])
            passed_indices = [i for i, ok in enumerate(passed) if ok and i < len(df)]
            result_df = df.iloc[passed_indices].reset_index(drop=True)
            logger.info(f"CNS strict filter: {len(df)} → {len(result_df)}")
            return result_df
        except Exception as e:
            logger.warning(f"CNS strict filter failed: {e}")
            return df

    # ============================================================
    # CNS 宽松过滤（旧接口保留）
    # ============================================================

    def is_cns_compliant(self, smiles: str) -> bool:
        """单分子 CNS 药物性质检测"""
        script = f"""
import json
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
mol = Chem.MolFromSmiles({smiles!r})
if mol is None:
    print(json.dumps({{"ok": False}}))
else:
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    ok = (mw <= 450) and (1.0 <= logp <= 4.5) and (tpsa <= 90) and (hbd <= 2) and (hba <= 7)
    print(json.dumps({{"ok": ok}}))
"""
        try:
            output = self._run_python("rdkit", script)
            return json.loads(output.strip()).get("ok", False)
        except Exception:
            return False

    # ============================================================
    # CNS + PAINS 批量过滤
    # ============================================================

    def apply_cns_pains_filter(
        self,
        df: pd.DataFrame,
        smiles_column: str = "SMILES",
    ) -> pd.DataFrame:
        """
        批量 CNS + PAINS 过滤

        Args:
            df: 输入 DataFrame
            smiles_column: SMILES 列名

        Returns:
            过滤后的 DataFrame
        """
        smiles_list = df[smiles_column].tolist() if smiles_column in df.columns else []
        if not smiles_list:
            return df

        smiles_json = json.dumps(smiles_list)

        script = f"""
import json
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, FilterCatalog

smiles_list = json.loads({smiles_json!r})

fparams = FilterCatalog.FilterCatalogParams()
fparams.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
pains = FilterCatalog.FilterCatalog(fparams)

results = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        results.append(False); continue
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        cns_ok = (mw <= 450) and (1.0 <= logp <= 4.5) and (tpsa <= 90) and (hbd <= 2) and (hba <= 7)
    except:
        cns_ok = False
    pains_ok = not pains.HasMatch(mol)
    results.append(cns_ok and pains_ok)

print(json.dumps({{"passed": results}}))
"""
        try:
            output = self._run_python("rdkit", script)
            passed = json.loads(output.strip()).get("passed", [])
            passed_idx = [i for i, ok in enumerate(passed) if ok]
            result_df = df.iloc[passed_idx].reset_index(drop=True) if passed_idx else pd.DataFrame()
            logger.info(f"CNS/PAINS: {len(df)} → {len(result_df)}")
            return result_df
        except Exception as e:
            logger.warning(f"CNS/PAINS filter failed: {e}")
            return df

    # ============================================================
    # Butina 聚类
    # ============================================================

    def cluster_molecules_butina(
        self,
        smiles_list: List[str],
        similarity_threshold: float = 0.7,
    ) -> List[List[int]]:
        """
        Butina Tanimoto 聚类

        Args:
            smiles_list: SMILES 列表
            similarity_threshold: Tanimoto 相似度阈值

        Returns:
            聚类结果: [[idx1, idx2, ...], [idx5, idx7, ...], ...]
        """
        if len(smiles_list) < 2:
            return [[i] for i in range(len(smiles_list))]

        smiles_json = json.dumps(smiles_list)

        script = f"""
import json
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina

smiles_list = json.loads({smiles_json!r})
threshold = {similarity_threshold}

mols = [Chem.MolFromSmiles(s) for s in smiles_list]
mols = [m for m in mols if m is not None]

if len(mols) < 2:
    print(json.dumps({{"clusters": [[i] for i in range(len(mols))]}}))
else:
    fgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [fgen.GetFingerprint(m) for m in mols]
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = BulkTanimotoSimilarity(fps[i], fps[:i])
        for s in sims:
            dists.append(1.0 - s)
    clusters = Butina.ClusterData(dists, n, 1.0 - threshold, isDistData=True, reordering=True)
    print(json.dumps({{"clusters": [list(c) for c in clusters]}}))
"""
        try:
            output = self._run_python("rdkit", script)
            return json.loads(output.strip()).get("clusters", [])
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return [[i] for i in range(len(smiles_list))]

    # ============================================================
    # 双受体选择性分析
    # ============================================================

    @staticmethod
    def analyze_dual_target_selectivity(
        df: pd.DataFrame,
        target1_col: str = "target1_energy",
        target2_col: str = "target2_energy",
    ) -> pd.DataFrame:
        """
        双受体选择性分析

        计算选择性比值，通过统计学方法筛选高选择性化合物。

        Args:
            df: 包含双靶点对接结果的 DataFrame
            target1_col: 主靶点结合能列名
            target2_col: 负向靶点结合能列名

        Returns:
            添加了 is_hit, Class_1/2/3 列的 DataFrame
        """
        df = df.dropna(subset=[target1_col, target2_col]).copy()
        df[target1_col] = pd.to_numeric(df[target1_col], errors='coerce')
        df[target2_col] = pd.to_numeric(df[target2_col], errors='coerce')
        df = df.dropna(subset=[target1_col, target2_col])

        selectivity_col = "selectivity_ratio"
        df[selectivity_col] = (
            df[target1_col] / df[target2_col].replace(0, np.nan)
        )

        t1_mean, t1_sd = df[target1_col].mean(), df[target1_col].std()
        sel_mean, sel_sd = df[selectivity_col].mean(), df[selectivity_col].std()

        cond_sel_high = df[selectivity_col] > (sel_mean + sel_sd)
        cond_sel_ultra = df[selectivity_col] > (sel_mean + 3 * sel_sd)
        cond_t1_high = df[target1_col] < (t1_mean - t1_sd)
        cond_t1_ultra = df[target1_col] < (t1_mean - 3 * t1_sd)
        cond_special = (df[target1_col] < -4.0) & (df[target2_col] > 0)
        cond_exclude = df[target1_col] > 0

        df["Class_1"] = cond_t1_high & (cond_sel_high | cond_special) & (~cond_exclude)
        df["Class_2"] = cond_t1_ultra & (~cond_exclude)
        df["Class_3"] = (cond_sel_ultra | cond_special) & (~cond_exclude)
        df["is_hit"] = df["Class_1"] | df["Class_2"] | df["Class_3"]

        logger.info(f"Dual-target: {df['is_hit'].sum()}/{len(df)} hits")
        return df

    # ============================================================
    # 种子挑选
    # ============================================================

    # ============================================================
    # 能量截断工具
    # ============================================================

    @staticmethod
    def clamp_energy(series: pd.Series, upper: float = 0.0, lower: float = -20.0) -> pd.Series:
        """
        截断结合能：
        - > 0 → 0（不结合，不区分正值大小）
        - < -20 → -20（防止极端值影响统计）
        - 保留原始值在 raw_energy 列
        """
        return series.clip(upper=upper, lower=lower)

    # ============================================================
    # 种子挑选
    # ============================================================
    # 多样性优先种子选择（自适应 Butina 阈值）
    # ============================================================

    def select_seeds_diverse(
        self,
        df: pd.DataFrame,
        num_seeds: int = 50,
        similarity_start: float = 0.4,
        similarity_max: float = 0.7,
        energy_column: str = "delta",
    ) -> pd.DataFrame:
        """
        多样性优先挑选 — 自适应 Butina 聚类

        从 0.4 开始聚类，簇数不够就逐步放宽到 0.5→0.6→max。
        每簇选最优分子，确保化学空间覆盖 + 恰好 num_seeds 个。

        Returns:
            种子 DataFrame（恰好 num_seeds 个）
        """
        if df.empty:
            return df

        smiles_list = df["SMILES"].tolist() if "SMILES" in df.columns else []
        if not smiles_list:
            return self._top_n(df, num_seeds)

        energies = df[energy_column].values if energy_column in df.columns else None

        threshold = similarity_start
        selected = []

        while threshold <= similarity_max + 0.01:
            clusters = self.cluster_molecules_butina(smiles_list, threshold)
            n_clusters = len(clusters)

            selected = []
            for cluster in clusters:
                best = min(cluster, key=lambda i: energies[i]) if energies is not None else list(cluster)[0]
                selected.append(best)

            logger.info(
                f"  Diversity sim={threshold:.1f}: "
                f"{n_clusters} clusters → {len(selected)} reps"
            )

            if len(selected) >= num_seeds:
                selected = selected[:num_seeds]
                break
            threshold += 0.05

        # 不够补满
        if len(selected) < num_seeds and energies is not None:
            ranked = sorted(range(len(energies)), key=lambda i: energies[i])
            for i in ranked:
                if i not in selected:
                    selected.append(i)
                if len(selected) >= num_seeds:
                    break

        selected = selected[:num_seeds]
        seeds = df.iloc[selected].copy()
        seeds["selection_reason"] = f"diverse_sim{threshold:.1f}"
        return seeds

    # ============================================================

    def select_seeds(
        self,
        docking_df: pd.DataFrame,
        num_seeds: Optional[int] = None,
        strategy: Optional[str] = None,
        round_num: int = 1,
        energy_column: str = "target_energy",
        anti_mode: bool = False,
    ) -> pd.DataFrame:
        """
        双靶点统计选择性筛选：

        selectivity = NMDA / Opioid

        入选三准则:
          1. selectivity > μ + 3σ  (超高选择性)
          2. NMDA < μ - 3σ          (超高靶点亲和力)
          3. selectivity > μ + σ AND NMDA < μ - σ  (高选择性+高亲和)

        剔除: NMDA > 0 或 Opioid > 0（正值=不结合，不含在统计中）
        多样性: Butina sim < 0.4 起始，不够 50 则自适应放宽
        """
        if docking_df.empty:
            return docking_df

        num_seeds = num_seeds or self.config["num_seeds"]
        strategy = strategy or self.config["strategy"]

        valid = docking_df[docking_df.get("docking_success", True) == True].copy()
        if valid.empty:
            return valid

        has_anti = "anti_energy" in valid.columns and valid["anti_energy"].notna().any()

        if not has_anti:
            return self._top_n(valid.sort_values(energy_column), num_seeds)

        # ── 保留原始值用于统计 ──
        for col in ["target_energy", "anti_energy"]:
            if col in valid.columns:
                valid[f"{col}_raw"] = valid[col]

        nmda_r = valid["target_energy_raw"]
        opioid_r = valid["anti_energy_raw"]

        # ── 剔除正值（不结合的分子不参与统计）──
        stat_mask = (nmda_r < 0) & (opioid_r < 0)
        stat_pool = valid[stat_mask]

        if len(stat_pool) < 10:
            logger.warning(f"Too few valid molecules for statistics ({len(stat_pool)})")
            valid["selectivity"] = nmda_r / opioid_r.replace(0, np.nan)
            return self._top_n(valid[stat_mask].sort_values("selectivity", ascending=False), num_seeds)

        # ── 计算选择性和统计量 ──
        sel_values = stat_pool["target_energy_raw"] / stat_pool["anti_energy_raw"]
        nmda_values = stat_pool["target_energy_raw"]

        sel_mean, sel_sd = sel_values.mean(), sel_values.std()
        nmda_mean, nmda_sd = nmda_values.mean(), nmda_values.std()

        logger.info(
            f"  Selectivity stats: μ={sel_mean:.2f}, σ={sel_sd:.2f} | "
            f"NMDA stats: μ={nmda_mean:.2f}, σ={nmda_sd:.2f}"
        )

        # ── 三准则筛选 ──
        criterion_1 = sel_values > (sel_mean + 3 * sel_sd)        # 超高选择性
        criterion_2 = nmda_values < (nmda_mean - 3 * nmda_sd)     # 超高靶点亲和力
        criterion_3 = (sel_values > (sel_mean + sel_sd)) & \
                       (nmda_values < (nmda_mean - nmda_sd))       # 高选择性+高亲和力

        hit_mask = criterion_1 | criterion_2 | criterion_3
        n_hits = hit_mask.sum()

        logger.info(
            f"  Statistical hits: C1(ultra-sel)={criterion_1.sum()} "
            f"C2(ultra-nmda)={criterion_2.sum()} "
            f"C3(high-both)={criterion_3.sum()} → {n_hits} total"
        )

        if n_hits == 0:
            logger.warning("No statistical hits — falling back to top selectivity")
            valid["selectivity"] = nmda_r / opioid_r.replace(0, np.nan)
            valid["delta"] = nmda_r - opioid_r
            top = valid[stat_mask].sort_values("selectivity", ascending=False).head(num_seeds)
            return top

        valid_hits = stat_pool[hit_mask].copy()
        valid_hits["selectivity"] = (
            valid_hits["target_energy_raw"] / valid_hits["anti_energy_raw"]
        )
        valid_hits["delta"] = (
            valid_hits["target_energy_raw"] - valid_hits["anti_energy_raw"]
        )

        # ── CNS 严格物性过滤 (MW<360, cLogP 2~4, etc.) ──
        cns_filtered = self.apply_cns_strict_filter(valid_hits)
        if not cns_filtered.empty:
            valid_hits = cns_filtered
        else:
            logger.warning("No CNS strict survivors — keeping all hits")

        # ── Lipinski + PAINS 过滤 ──
        filtered = self.apply_cns_pains_filter(valid_hits)
        if filtered.empty:
            logger.warning("No CNS/PAINS survivors among hits")
            filtered = valid_hits

        # ── 多样性聚类（sim < 0.4，不够则自适应放宽到 0.7）──
        seeds = self.select_seeds_diverse(
            filtered,
            num_seeds=num_seeds,
            similarity_start=0.4,
            similarity_max=0.7,
            energy_column="selectivity",
        )

        logger.info(
            f"[Round {round_num}] {len(seeds)} seeds "
            f"(NMDA: {seeds['target_energy_raw'].min():.1f}~{seeds['target_energy_raw'].max():.1f}, "
            f"Opioid: {seeds['anti_energy_raw'].min():.1f}~{seeds['anti_energy_raw'].max():.1f}, "
            f"selectivity: {seeds['selectivity'].min():.1f}~{seeds['selectivity'].max():.1f})"
        )
        return seeds

    def _top_n(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        seeds = df.head(min(n, len(df))).copy()
        seeds["selection_reason"] = f"top_{len(seeds)}"
        return seeds

    def _by_threshold(
        self, df: pd.DataFrame, energy_col: str, n: int
    ) -> pd.DataFrame:
        threshold = self.config["binding_energy_threshold"]
        seeds = df[df[energy_col] <= threshold].copy()
        if len(seeds) == 0:
            logger.warning(f"No compounds below {threshold}, using top {n}")
            return self._top_n(df, n)
        if len(seeds) > n * 2:
            seeds = seeds.head(n * 2)
        seeds["selection_reason"] = f"energy<={threshold}"
        return seeds

    def _by_clustering(
        self, df: pd.DataFrame, energy_col: str, n: int
    ) -> pd.DataFrame:
        smiles_list = df["SMILES"].tolist() if "SMILES" in df.columns else []
        if not smiles_list:
            return self._top_n(df, n)

        sim = self.config.get("cluster_similarity", 0.7)
        clusters = self.cluster_molecules_butina(smiles_list, sim)
        logger.info(f"{len(smiles_list)} molecules → {len(clusters)} clusters")

        # 每簇挑能量最优的
        selected = []
        energies = df[energy_col].values if energy_col in df.columns else None
        for cluster in clusters:
            if energies is not None:
                best = min(cluster, key=lambda i: energies[i])
            else:
                best = list(cluster)[0]
            selected.append(best)

        # 不够则补充
        if len(selected) < n and energies is not None:
            all_sorted = sorted(range(len(energies)), key=lambda i: energies[i])
            for idx in all_sorted:
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= n:
                    break

        seeds = df.iloc[selected[:n]].copy()
        seeds["selection_reason"] = f"cluster_sim_{sim}"
        return seeds

    # ============================================================
    # 导出
    # ============================================================

    def export_seeds_to_csv(
        self, seeds: pd.DataFrame, output_path: Path,
        smiles_column: str = "SMILES",
    ) -> Path:
        """导出种子为 CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [smiles_column] if smiles_column in seeds.columns else seeds.columns
        seeds[cols].to_csv(output_path, index=False)
        logger.info(f"Exported {len(seeds)} seeds → {output_path}")
        return output_path

    def export_seeds_to_smi(
        self, seeds: pd.DataFrame, output_path: Path,
        smiles_column: str = "SMILES",
    ) -> Path:
        """导出种子为 SMI 文件（每行一个 SMILES）"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for _, row in seeds.iterrows():
                smi = row.get(smiles_column, "")
                if smi:
                    f.write(f"{smi}\n")
        logger.info(f"Exported {len(seeds)} seeds → {output_path}")
        return output_path


# ============================================================
# 兼容性别名：PostScreeningClustering 功能已合并
# ============================================================

class PostScreeningClustering(SeedSelector):
    """
    筛选后聚类分析（兼容性别名，功能已合并到 SeedSelector）

    提供与旧代码兼容的接口。
    """

    def cluster_docking_results(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        round_num: int,
        smiles_column: str = "SMILES",
        energy_column: str = "binding_energy",
        similarity_threshold: float = 0.7,
    ) -> pd.DataFrame:
        """对接结果聚类分析（兼容接口）"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        valid = df[df.get("docking_success", True) == True].copy()
        if valid.empty or smiles_column not in valid.columns:
            df["cluster_id"] = -1
            return df

        smiles_list = valid[smiles_column].tolist()
        clusters = self.cluster_molecules_butina(smiles_list, similarity_threshold)

        cluster_map = {}
        for cid, members in enumerate(clusters):
            for mi in members:
                cluster_map[valid.index[mi]] = cid

        valid["cluster_id"] = valid.index.map(lambda x: cluster_map.get(x, -1))

        # 保存
        valid.to_csv(output_dir / f"clustering_round{round_num}.csv", index=False)

        stats = valid.groupby("cluster_id").agg(
            count=("cluster_id", "size"),
            mean_energy=(energy_column, "mean"),
            best_energy=(energy_column, "min"),
        ).reset_index()
        stats.to_csv(output_dir / f"cluster_stats_round{round_num}.csv", index=False)

        return valid


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    selector = SeedSelector()
    print(f"Strategy: {selector.config['strategy']}")
    print(f"Num seeds: {selector.config['num_seeds']}")

    # 快速聚类测试
    test_smiles = [
        "CCO", "CCN", "CCC", "CCCl",
        "c1ccccc1", "c1ccccc1O", "c1ccccc1N",
    ]
    clusters = selector.cluster_molecules_butina(test_smiles, 0.5)
    print(f"Clusters: {len(clusters)}")
    for i, c in enumerate(clusters):
        print(f"  Cluster {i}: {c}")

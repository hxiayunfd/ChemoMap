#!/usr/bin/env python3
"""
模块: REINVENT4 化合物库生成器 (Mol2Mol)
Module: REINVENT4 Compound Library Generator (Mol2Mol)

基于种子分子，通过 REINVENT4 mol2mol_medium_similarity.prior 模型
进行相似性采样生成新分子库。

工作流:
1. 每个种子 → 写入临时 .smi 文件
2. 生成 TOML 配置 → 调用 reinvent CLI
3. 解析输出 CSV → 分子命名: {seed_name}_r{round}_{index}
4. 合并所有种子生成的分子 → 返回统一 DataFrame

用法:
    gen = ReinventGenerator()
    df = gen.generate_from_seeds(
        seed_df,           # DataFrame with SMILES column
        output_dir=Path("round_002/01_generation"),
        round_num=2,
        num_per_seed=40,
    )
"""

import csv
import io
import logging
import sys
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import REINVENT_CONFIG, DIRS

logger = logging.getLogger(__name__)


# ── REINVENT4 工具路径 ──
from config import TOOLS as _CFG_TOOLS, MOL2MOL_PRIOR_PATH

REINVENT_BIN = _CFG_TOOLS.get("reinvent4", "reinvent")
MOL2MOL_PRIOR = MOL2MOL_PRIOR_PATH


class ReinventGenerator:
    """
    REINVENT4 Mol2Mol 化合物生成器

    基于种子 SMILES 进行相似性采样，生成新分子库。
    当 REINVENT4 不可用时自动降级为模拟模式。
    """

    def __init__(self, config: Optional[Dict] = None, conda_manager=None):
        self.config = {**REINVENT_CONFIG, **(config or {})}
        self._mock_mode = False

        if conda_manager is None:
            from conda_manager import get_conda_manager
            conda_manager = get_conda_manager()
        self.cm = conda_manager

        self._check_tool()

    def _check_tool(self):
        """检测 REINVENT4 是否可用"""
        reinvent = Path(REINVENT_BIN)
        prior = Path(MOL2MOL_PRIOR)

        if not reinvent.exists():
            self._enable_mock(f"reinvent not found: {REINVENT_BIN}")
            return
        if not prior.exists():
            self._enable_mock(f"prior not found: {MOL2MOL_PRIOR}")
            return

        logger.info(
            f"REINVENT4 ready: {REINVENT_BIN}\n"
            f"  Prior: {MOL2MOL_PRIOR}"
        )

    def _enable_mock(self, reason: str):
        self._mock_mode = True
        logger.warning(f"REINVENT4 mock mode: {reason}")

    # ============================================================
    # TOML 生成
    # ============================================================

    def _write_toml(
        self,
        seed_smi_path: Path,
        output_file: Path,
        round_num: int,
        total_smiles: int,
        temperature: float = 0.8,
        strategy: str = "multinomial",
    ) -> Path:
        """
        写入 REINVENT4 Mol2Mol TOML 配置文件

        Args:
            seed_smi_path: 种子 SMILES 文件（每行一个）
            output_file: 输出 CSV 路径
            round_num: 轮次
            num_smiles: 生成的分子数
            temperature: 采样温度
            strategy: 采样策略 (multinomial / beamsearch)

        Returns:
            TOML 文件路径
        """
        output_dir = output_file.parent
        toml_path = output_dir / f"reinvent_r{round_num}.toml"

        content = f"""# REINVENT4 Mol2Mol Sampling — Round {round_num}
run_type = "sampling"
device = "cuda:0"
json_out_config = "_sampling_r{round_num}.json"

[parameters]
model_file = "{MOL2MOL_PRIOR}"
smiles_file = "{seed_smi_path}"

sample_strategy = "{strategy}"
temperature = {temperature}

output_file = "{output_file.name}"

num_smiles = {total_smiles}
unique_molecules = true
randomize_smiles = false
"""
        toml_path.write_text(content)
        logger.debug(f"TOML written: {toml_path}")
        return toml_path

    def _write_seed_smi(
        self, seeds: pd.DataFrame, smi_path: Path, round_num: int,
    ) -> pd.DataFrame:
        """
        写入种子 SMILES + 构建命名映射

        命名规则（家族制，不长）:
          - 第一轮: {family_prefix}_R1_{idx}   (如 K01_R1_003)
          - 后续轮: {family_prefix}_R{round}_{idx}  (如 K01_R2_005)
          - family_prefix 跨轮不变，编码遗传来源

        lineage 列存储完整进化路径:
          K01:ketamine → K01_R2_3 → K01_R3_1

        Returns:
            添加了 family, seed_name 列的种子 DataFrame
        """
        seeds = seeds.copy()
        families = []
        seed_names = []
        lineages = []

        for i, (_, row) in enumerate(seeds.iterrows()):
            # 确定家族 ID
            if "family" in row and pd.notna(row["family"]):
                family = str(row["family"])
            elif "Compound" in row and pd.notna(row["Compound"]):
                # 从名字提取家族：K01_R2_003 → K01
                compound = str(row["Compound"])
                parts = compound.split("_")
                if len(parts) >= 2 and parts[0].startswith("K"):
                    family = parts[0]
                else:
                    # 第一轮种子
                    family = f"K{i+1:02d}"
            else:
                family = f"K{i+1:02d}"

            seed_name = f"{family}_R{round_num}"

            # 构建 lineage
            if "lineage" in row and pd.notna(row["lineage"]):
                lineage = f"{row['lineage']} → {seed_name}"
            else:
                lineage = f"{family}:{str(row.get('Compound', row.get('SMILES', '')))[:30]} → {seed_name}"

            families.append(family)
            seed_names.append(seed_name)
            lineages.append(lineage)

        seeds["family"] = families
        seeds["seed_name"] = seed_names
        seeds["generation"] = round_num  # 第几代进化

        # REINVENT4 的 smiles_file 格式: SMILES\tname
        with open(smi_path, "w") as f:
            for name, smi in zip(seed_names, seeds["SMILES"]):
                f.write(f"{smi}\t{name}\n")

        logger.info(
            f"Seed SMILES: {smi_path} ({len(families)} seeds, "
            f"families: {list(dict.fromkeys(families))})"
        )
        return seeds

    # ============================================================
    # 生成
    # ============================================================

    def generate(
        self,
        output_dir: Path,
        seed_smiles: Optional[List[str]] = None,
        round_num: int = 1,
        timeout: int = None,
    ) -> pd.DataFrame:
        """
        运行 REINVENT4 生成化合物库

        Args:
            output_dir: 输出目录
            seed_smiles: 种子 SMILES 列表
            round_num: 轮次
            timeout: 超时时间 (秒)

        Returns:
            DataFrame: SMILES, score, round, seed_name
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._mock_mode or seed_smiles is None:
            return self._generate_mock(round_num, seed_smiles)

        # 创建种子 DataFrame
        seed_df = pd.DataFrame({"SMILES": seed_smiles})

        return self.generate_from_seeds(
            seed_df=seed_df,
            output_dir=output_dir,
            round_num=round_num,
            **({"timeout": timeout} if timeout else {}),
        )

    def generate_from_seeds(
        self,
        seed_df: pd.DataFrame,
        output_dir: Path,
        round_num: int = 1,
        num_per_seed: Optional[int] = None,
        temperature: float = 0.8,
        timeout: int = None,
        conda_env: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        从种子 DataFrame 生成分子库

        每个种子独立运行 REINVENT4，生成命名分子后合并。

        Args:
            seed_df: 种子 DataFrame (需含 SMILES 列)
            output_dir: 输出目录
            round_num: 轮次
            num_per_seed: 每个种子生成的分子数 (默认: total / n_seeds)
            temperature: 采样温度 (0.5-1.0, 越低越保守)
            timeout: 单个种子生成超时 (秒)
            conda_env: REINVENT4 conda 环境

        Returns:
            DataFrame: SMILES, score, round, seed_name, source_seed
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._mock_mode:
            smi_list = seed_df["SMILES"].tolist() if "SMILES" in seed_df.columns else []
            return self._generate_mock(round_num, smi_list)

        n_seeds = len(seed_df)
        if n_seeds == 0:
            logger.warning("No seeds provided")
            return pd.DataFrame()
        # num_smiles = 每个种子生成数（不是总数），500/seed × 50 seeds = 25000
        num_per_seed = num_per_seed or self.config.get("num_per_seed") or 500
        total_smiles = num_per_seed * n_seeds
        logger.info(
            f"[Round {round_num}] Mol2Mol sampling: "
            f"{n_seeds} seeds × {num_per_seed}/seed = {total_smiles} total"
        )
        logger.info(
            f"  ⏳ REINVENT4 is generating (GPU). "
            f"Check 'nvidia-smi' — if GPU util > 50% it's working. "
            f"Est. ~{max(1, total_smiles // 800)} min for this round."
        )

        # 创建命名
        seed_df = self._write_seed_smi(seed_df, output_dir / f"seeds_r{round_num}.smi", round_num)
        families = seed_df["family"].tolist()
        seed_names = seed_df["seed_name"].tolist()
        seed_smiles = seed_df["SMILES"].tolist()
        seed_compounds = seed_df.get("Compound", pd.Series([""] * n_seeds)).tolist()
        lineages = seed_df.get("lineage", pd.Series([""] * n_seeds)).tolist()
        pedigrees = seed_df.get("pedigree", pd.Series(families)).tolist()

        conda_env = conda_env or self.cm.get_env("reinvent4")
        all_dfs = []

        # ── 逐种子生成（每个种子完成后打印进度）──
        for si in range(n_seeds):
            seed_name = seed_names[si]
            family = families[si]
            smi = seed_smiles[si]

            seed_smi_path = output_dir / f"seed_{si:03d}.smi"
            seed_smi_path.write_text(f"{smi}\t{seed_name}\n")
            seed_csv = output_dir / f"gen_seed{si:03d}.csv"

            toml_path = self._write_toml(
                seed_smi_path, seed_csv, round_num,
                total_smiles=num_per_seed, temperature=temperature,
            )
            cmd = ["conda", "run", "-n", conda_env, str(REINVENT_BIN), str(toml_path)]

            try:
                result = subprocess.run(
                    cmd, cwd=str(output_dir),
                    capture_output=True, text=True,
                    **({"timeout": timeout} if timeout else {}),
                )
                if result.returncode != 0:
                    logger.error(f"  [{si+1}/{n_seeds}] {seed_name} FAILED: {result.stderr[-150:]}")
                    continue

                if seed_csv.exists():
                    one_seed = pd.DataFrame([{
                        "SMILES": smi, "family": family, "seed_name": seed_name,
                        "Compound": seed_compounds[si] if si < len(seed_compounds) else "",
                        "lineage": lineages[si] if si < len(lineages) else "",
                        "pedigree": pedigrees[si] if si < len(pedigrees) else family,
                    }])
                    df_seed = self._parse_output(seed_csv, one_seed, round_num)
                    all_dfs.append(df_seed)

            except subprocess.TimeoutExpired:
                logger.error(f"  [{si+1}/{n_seeds}] {seed_name} TIMED OUT")

            # 实时进度
            n = len(all_dfs[-1]) if all_dfs else 0
            sys.stdout.write(f"\r  [{si+1}/{n_seeds}] {seed_name} → {n} mols")
            sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()

        if not all_dfs:
            logger.warning("No molecules generated")
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

    # ============================================================
    # 结果解析 + 命名
    # ============================================================

    def _parse_output(
        self,
        output_csv: Path,
        seed_df: pd.DataFrame,
        round_num: int,
    ) -> pd.DataFrame:
        """
        解析 REINVENT4 输出 CSV，分配分子命名

        REINVENT4 Mol2Mol CSV 格式:
            SMILES,SMILES_state,Input_SMILES,Tanimoto,NLL
            CCO...,1,CNC1...,0.577,2.36

        命名规则: {seed_name}_{index}
        """
        if not output_csv.exists():
            logger.warning(f"Output not found: {output_csv}")
            return self._generate_mock(round_num, seed_df["SMILES"].tolist())

        # 用 pandas 读取 CSV（自动跳过 header）
        try:
            df_raw = pd.read_csv(output_csv)
        except Exception:
            logger.warning(f"Failed to parse CSV: {output_csv}")
            return self._generate_mock(round_num, seed_df["SMILES"].tolist())

        if df_raw.empty:
            logger.warning("Empty output CSV")
            return self._generate_mock(round_num, seed_df["SMILES"].tolist())

        # 查找 SMILES 列
        smiles_col = None
        for col in df_raw.columns:
            if col.strip().upper() == "SMILES":
                smiles_col = col
                break
        if smiles_col is None:
            smiles_col = df_raw.columns[0]  # 回退：第一列

        # 提取 Tanimoto/NLL 作为 score（如果有）
        score_col = None
        for col in df_raw.columns:
            col_upper = col.strip().upper()
            if col_upper in ("TANIMOTO", "NLL"):
                score_col = col
                break
        # 默认用 NLL 的倒数作为 score（越低越好 → 越高越好）
        nll_col = None
        for col in df_raw.columns:
            if col.strip().upper() == "NLL":
                nll_col = col
                break

        compounds = []
        for _, row in df_raw.iterrows():
            smi = str(row[smiles_col]).strip()
            if not smi or smi == "nan":
                continue
            score = 0.0
            if score_col is not None:
                try:
                    score = float(row[score_col])
                except (ValueError, TypeError):
                    score = 0.0
            elif nll_col is not None:
                try:
                    nll = float(row[nll_col])
                    score = round(1.0 / (1.0 + nll), 3) if nll > 0 else 1.0
                except (ValueError, TypeError):
                    score = 0.0
            compounds.append({
                "SMILES": smi,
                "score": score,
                "round": round_num,
            })

        if not compounds:
            logger.warning("No compounds in output")
            return self._generate_mock(round_num, seed_df["SMILES"].tolist())

        df = pd.DataFrame(compounds)

        # ── 分子命名 + 遗传信息 ──
        #   Compound:  {family}_R{round}_{idx:03d}    (短名，固定长度)
        #   parent:    上一轮种子的 Compound 名          (直接知道来自谁)
        #   pedigree:  K01>2:3>3:1                     (紧凑完整谱系)
        #   lineage:   K01:ketamine → K01_R2_003 → ... (人类可读史)
        # ─────────────────────────────────────────────
        seed_smiles_list = seed_df["SMILES"].tolist()
        seed_compounds = seed_df.get("Compound", pd.Series([""] * len(seed_df))).tolist()
        families = seed_df["family"].tolist()
        lineages = seed_df.get("lineage", pd.Series([""] * len(seed_df))).tolist()
        pedigrees = seed_df.get("pedigree", pd.Series(families)).tolist()

        num_per_seed = max(1, len(df) // len(seed_df)) if len(seed_df) > 0 else 0

        source_seeds = []
        mol_names = []
        mol_families = []
        mol_parents = []
        mol_pedigrees = []
        mol_lineages = []

        for i in range(len(df)):
            seed_idx = min(i // num_per_seed, len(seed_df) - 1)
            mol_idx = (i % num_per_seed) + 1

            family = families[seed_idx]
            mol_name = f"{family}_R{round_num}_{mol_idx:03d}"
            # 紧凑谱系: K01>2:3 表示 K01 家族第2轮第3个
            parent_pedigree = str(pedigrees[seed_idx]) if seed_idx < len(pedigrees) else family
            mol_pedigree = f"{parent_pedigree}>{round_num}:{mol_idx}"

            parent_name = seed_compounds[seed_idx] if seed_idx < len(seed_compounds) else ""
            parent_lineage = lineages[seed_idx] if seed_idx < len(lineages) else family

            source_seeds.append(seed_smiles_list[seed_idx])
            mol_names.append(mol_name)
            mol_families.append(family)
            mol_parents.append(parent_name)
            mol_pedigrees.append(mol_pedigree)
            mol_lineages.append(f"{parent_lineage} → {mol_name}")

        df["source_seed"] = source_seeds
        df["Compound"] = mol_names
        df["family"] = mol_families
        df["parent"] = mol_parents          # 直接父代种子名
        df["pedigree"] = mol_pedigrees      # 紧凑谱系: K01>2:3>3:1
        df["lineage"] = mol_lineages        # 人类可读进化史
        df["generation"] = round_num

        logger.info(
            f"[Round {round_num}] Parsed {len(df)} compounds "
            f"from {len(seed_df)} seeds ({len(df) // max(1, len(seed_df))}/seed)"
        )
        return df

    # ============================================================
    # 旧接口兼容
    # ============================================================

    def generate_from_seed_list(
        self,
        seed_smiles: List[str],
        output_dir: Path,
        round_num: int = 1,
    ) -> pd.DataFrame:
        """旧接口兼容：从 SMILES 列表生成"""
        seed_df = pd.DataFrame({"SMILES": seed_smiles})
        return self.generate_from_seeds(seed_df, output_dir, round_num)

    # ============================================================
    # 模拟模式（REINVENT4 不可用时）
    # ============================================================

    def _generate_mock(
        self,
        round_num: int,
        seed_smiles: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """生成模拟化合物数据（测试用）"""

        MOL_TEMPLATES = [
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
            "COC1=CC2=C(C=C1)C(=O)C(=CO2)C3=CC=C(C=C3)O",
            "CC(C)(C)NC(=O)C1=CC2=C(C=C1)OCCO2",
            "CN1CCN(CC1)C2=CC=C(C=C2)C(=O)NC3=CC=CC4=C3C=CC=C4",
            "C1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3)O2)O",
            "CC1=CC(=O)C2=C(C=C(C=C2)O1)O",
            "C1=CC=C2C(=C1)C(=O)C3=C(C2=O)C=CC=C3",
            "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",
            "COC1=CC=CC=C1OC2=CC=CC=C2",
            "C1=CC=C(C=C1)C(=O)N",
        ]

        n_per_seed = max(1, self.config["num_smiles"] // max(1, len(seed_smiles or [])))
        compounds = []

        if seed_smiles:
            for si, seed in enumerate(seed_smiles):
                family = f"K{si+1:02d}"
                for i in range(n_per_seed):
                    mol_name = f"{family}_R{round_num}_{i+1:03d}"
                    compounds.append({
                        "SMILES": random.choice(MOL_TEMPLATES + [seed]),
                        "score": round(random.uniform(0.3, 0.95), 3),
                        "round": round_num,
                        "source_seed": seed,
                        "Compound": mol_name,
                        "family": family,
                        "lineage": f"{family}:mock_seed → {mol_name}",
                        "generation": round_num,
                    })
        else:
            for i in range(self.config["num_smiles"]):
                mol_name = f"K00_R{round_num}_{i+1:03d}"
                compounds.append({
                    "SMILES": random.choice(MOL_TEMPLATES),
                    "score": round(random.uniform(0.3, 0.95), 3),
                    "round": round_num,
                    "source_seed": "",
                    "Compound": mol_name,
                    "family": "K00",
                    "lineage": f"de_novo → {mol_name}",
                    "generation": round_num,
                })

        df = pd.DataFrame(compounds)
        logger.info(
            f"[Round {round_num}] Generated {len(df)} MOCK compounds "
            f"(mol2mol mock)"
        )
        return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    gen = ReinventGenerator()
    print(f"Mock mode: {gen._mock_mode}")

    # 测试种子生成
    test_seeds = pd.DataFrame({
        "SMILES": [
            "CNC1(C2=CC=CC=C2Cl)CCCCC1=O",
            "CNC(CCCC(C4=O)C2=CSC(N3CCOCC3)=C2)4C1=CC=C(C=C1)C1CCC1F",
        ],
    })

    df = gen.generate_from_seeds(
        test_seeds,
        output_dir=DIRS["temp"] / "reinvent_test2",
        round_num=2,
        num_per_seed=10,
    )
    print(df[["Compound", "source_seed"]].head(12))
    print(f"Total: {len(df)}")

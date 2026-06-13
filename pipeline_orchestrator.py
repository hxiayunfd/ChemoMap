#!/usr/bin/env python3
"""
模块: 流水线编排器
Module: Pipeline Orchestrator

协调整个分子定向进化循环的各个模块。

完整流程 (7 步):
1. [Reinvent4]  生成化合物库
2. [RDKit]      理化性质筛选 (Lipinski/PAINS)
3. [RDKit/Meeko] 配体准备 (SMILES → 3D SDF → PDBQT)
4. [AutoDock-GPU] 分子对接
5. [RDKit]      结果筛选与种子挑选
6. [循环]       种子 → 下一轮 Reinvent4 生成
7. [收敛]       收敛检测与早停

用法:
    from pipeline_orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator(
        receptor_pdbqt="protein.pdbqt",
        center_x=10, center_y=25, center_z=12,
        size_x=30, size_y=30, size_z=30,
    )
    summaries = orch.run_iterative_screening(
        initial_seeds=["CCO", "CCN"],
        max_rounds=5,
    )
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    DIRS, ADGPU_CONFIG, SEED_SELECTION, ITERATION_CONFIG,
    create_directories,
)

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    虚拟筛选流水线编排器

    协调各模块完成迭代筛选循环。
    """

    def __init__(
        self,
        receptor_pdbqt: Optional[str] = None,
        center_x: float = 0.0,
        center_y: float = 0.0,
        center_z: float = 0.0,
        size_x: int = 30,
        size_y: int = 30,
        size_z: int = 30,
        gpu: int = 0,
        config_override: Optional[Dict] = None,
        conda_manager=None,
        anti_receptor_pdbqt: Optional[str] = None,
        anti_grid_params: Optional[tuple] = None,
    ):
        """
        Args:
            receptor_pdbqt: 主靶点受体 PDBQT
            center_x/y/z, size_x/y/z: 主靶点网格参数
            anti_receptor_pdbqt: 负靶点受体 PDBQT（None = 单靶点模式）
            anti_grid_params: (cx, cy, cz, sx, sy, sz) 负靶点网格参数
        """
        create_directories()

        if conda_manager is None:
            from conda_manager import get_conda_manager
            conda_manager = get_conda_manager()
        self.cm = conda_manager

        # 对接配置
        self.docking_config = {
            "receptor": receptor_pdbqt or ADGPU_CONFIG.get("receptor", ""),
            "center_x": center_x, "center_y": center_y, "center_z": center_z,
            "size_x": size_x, "size_y": size_y, "size_z": size_z,
            "gpu": gpu,
        }
        if config_override:
            self.docking_config.update(config_override.get("docking", {}))

        # 负靶点（anti-target）配置
        self.anti_enabled = anti_receptor_pdbqt is not None
        self._anti_receptor = anti_receptor_pdbqt
        self._anti_grid = anti_grid_params  # (cx, cy, cz, sx, sy, sz)

        # 惰性初始化
        self._generator = None
        self._filter = None
        self._preparator = None
        self._docking_engine = None
        self._anti_docking_engine = None
        self._selector = None

        # 迭代状态
        self.current_round: int = 0
        self.history: List[Dict] = []
        self.best_scores: List[float] = []
        self._last_seeds_df: Optional[pd.DataFrame] = None

        mode = "dual-target" if self.anti_enabled else "single-target"
        logger.info(f"PipelineOrchestrator initialized ({mode} mode)")

    # ============================================================
    # 惰性模块加载
    # ============================================================

    @property
    def generator(self):
        if self._generator is None:
            from reinvent_generator import ReinventGenerator
            self._generator = ReinventGenerator(conda_manager=self.cm)
        return self._generator

    @property
    def filter(self):
        if self._filter is None:
            from property_filter import PropertyFilter
            self._filter = PropertyFilter(conda_manager=self.cm)
        return self._filter

    @property
    def preparator(self):
        if self._preparator is None:
            from ligand_preparator import LigandPreparator
            self._preparator = LigandPreparator(conda_manager=self.cm)
        return self._preparator

    @property
    def anti_docking_engine(self):
        if self._anti_docking_engine is None and self.anti_enabled:
            from docking_engine import DockingEngine
            acx, acy, acz, asx, asy, asz = self._anti_grid
            self._anti_docking_engine = DockingEngine(
                config={
                    "receptor": self._anti_receptor,
                    "center_x": acx, "center_y": acy, "center_z": acz,
                    "size_x": asx, "size_y": asy, "size_z": asz,
                    "gpu": self.docking_config.get("gpu", 0),
                },
                conda_manager=self.cm,
            )
        return self._anti_docking_engine

    @property
    def docking_engine(self):
        if self._docking_engine is None:
            from docking_engine import DockingEngine
            self._docking_engine = DockingEngine(
                config=self.docking_config,
                conda_manager=self.cm,
            )
        return self._docking_engine

    @property
    def selector(self):
        if self._selector is None:
            from seed_selector import SeedSelector
            self._selector = SeedSelector(conda_manager=self.cm)
        return self._selector

    # ============================================================
    # 单轮运行
    # ============================================================

    def run_round(
        self,
        seed_smiles: Optional[List[str]] = None,
        round_num: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        运行一轮完整虚拟筛选

        Args:
            seed_smiles: 种子 SMILES 列表
            round_num: 轮次编号

        Returns:
            本轮结果摘要
        """
        if round_num is not None:
            self.current_round = round_num
        else:
            self.current_round += 1
        rn = self.current_round

        round_dir = DIRS["rounds"] / f"round_{rn:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"  Round {rn}")
        logger.info(f"{'='*60}")

        # ── Step 1: 生成化合物 ──
        gen_dir = round_dir / "01_generation"
        gen_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[Step 1/6] Generating compounds (Mol2Mol sampling)...")

        try:
            if seed_smiles:
                # 从种子 SMILES 列表构建 DataFrame
                seed_df = pd.DataFrame({"SMILES": seed_smiles})
                # 传递上一轮的家族、命名、谱系信息
                if self._last_seeds_df is not None:
                    for col in ["Compound", "family", "lineage", "pedigree", "generation"]:
                        if col in self._last_seeds_df.columns:
                            vals = self._last_seeds_df[col].values
                            seed_df[col] = list(vals[:len(seed_df)]) + [""] * max(0, len(seed_df) - len(vals))
            else:
                seed_df = pd.DataFrame({"SMILES": seed_smiles or []})

            compounds_df = self.generator.generate_from_seeds(
                seed_df=seed_df,
                output_dir=gen_dir,
                round_num=rn,
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._fail(rn, "generation_error", str(e))

        if compounds_df.empty:
            return self._fail(rn, "empty_generation")

        logger.info(f"  Generated {len(compounds_df)} compounds")

        # ── Step 2: 理化性质筛选 ──
        filter_dir = round_dir / "02_filtered"
        filter_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[Step 2/6] Filtering properties...")

        try:
            filtered_df = self.filter.filter_dataframe(compounds_df)
        except Exception as e:
            logger.error(f"Property filter failed: {e}")
            filtered_df = compounds_df

        filtered_csv = filter_dir / f"filtered_round{rn}.csv"
        filtered_df.to_csv(filtered_csv, index=False)
        logger.info(f"  {len(filtered_df)} passed filters")

        if filtered_df.empty:
            return self._fail(rn, "no_compounds_after_filter")

        # ── Step 3: 配体准备 (SMILES → PDBQT) ──
        pdbqt_dir = round_dir / "03_pdbqt"
        pdbqt_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[Step 3/6] Preparing ligands (SMILES → PDBQT)...")

        try:
            pdbqt_df = self.preparator.batch_convert_to_pdbqt(filtered_df, pdbqt_dir)
        except Exception as e:
            logger.error(f"Ligand preparation failed: {e}")
            return self._fail(rn, "ligand_prep_error", str(e))

        valid_pdbqt = pdbqt_df[pdbqt_df["pdbqt_exists"] == True]
        logger.info(f"  {len(valid_pdbqt)} PDBQT ligands ready")

        if valid_pdbqt.empty:
            return self._fail(rn, "no_pdbqt")

        # ── Step 4: 分子对接（主靶点 + 负靶点）──
        dock_dir = round_dir / "04_docking"
        dock_dir.mkdir(parents=True, exist_ok=True)
        anti_dock_dir = round_dir / "04b_anti_docking"

        if self.anti_enabled:
            anti_dock_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[Step 4/6] Docking against TARGET + ANTI-TARGET...")
        else:
            logger.info("[Step 4/6] AutoDock-GPU docking (single target)...")

        # 主靶点对接
        try:
            docking_results = self.docking_engine.batch_dock(valid_pdbqt, dock_dir)
            docking_results.rename(
                columns={"binding_energy": "target_energy",
                         "docking_success": "target_dock_ok"},
                inplace=True,
            )
        except Exception as e:
            logger.error(f"Target docking failed: {e}")
            return self._fail(rn, "docking_error", str(e))

        # 负靶点对接
        if self.anti_enabled:
            try:
                anti_results = self.anti_docking_engine.batch_dock(
                    valid_pdbqt, anti_dock_dir,
                )
                # 合并：只取负靶点的结合能
                docking_results["anti_energy"] = anti_results["binding_energy"].values
                docking_results["anti_dock_ok"] = anti_results["docking_success"].values
            except Exception as e:
                logger.error(f"Anti-target docking failed: {e}")
                docking_results["anti_energy"] = None
                docking_results["anti_dock_ok"] = False
        else:
            docking_results["anti_energy"] = None
            docking_results["anti_dock_ok"] = False

        # ── 能量截断（clamping）──
        # 正值=不结合，截断到 0 避免极端正值影响统计
        # 负靶点：>0 表示不结合（好），但不过度区分大正值
        for col in ["target_energy", "anti_energy"]:
            if col in docking_results.columns:
                mask = docking_results[col].notna()
                docking_results.loc[mask, f"{col}_raw"] = docking_results.loc[mask, col]
                # 截断：> 0 → 0；< -20 → -20（防止极端负值）
                docking_results.loc[mask, col] = docking_results.loc[mask, col].clip(upper=0.0, lower=-20.0)

        # 计算选择性 delta = target - anti（负值=靶点选择性好）
        if self.anti_enabled:
            docking_results["delta"] = (
                docking_results["target_energy"] - docking_results["anti_energy"]
            )
        else:
            docking_results["delta"] = 0.0
            docking_results["binding_energy"] = docking_results["target_energy"]

        # 统一 docking_success 标志
        if self.anti_enabled:
            docking_results["docking_success"] = (
                docking_results["target_dock_ok"] & docking_results["anti_dock_ok"]
            )
        else:
            docking_results["docking_success"] = docking_results["target_dock_ok"]

        docking_csv = dock_dir / f"docking_results_round{rn}.csv"
        docking_results.to_csv(docking_csv, index=False)

        n_ok = (docking_results["docking_success"] == True).sum()
        logger.info(f"  Docking: {n_ok}/{len(docking_results)} success")

        if n_ok == 0:
            return self._fail(rn, "all_docking_failed")

        # ── Step 5: 结果分析与种子挑选 ──
        analysis_dir = round_dir / "05_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[Step 5/6] Selecting seeds...")

        try:
            seeds = self.selector.select_seeds(
                docking_results,
                round_num=rn,
            )
        except Exception as e:
            logger.error(f"Seed selection failed: {e}")
            # 使用原始 top_n 作为回退
            valid_results = docking_results[docking_results["docking_success"] == True]
            seeds = valid_results.head(SEED_SELECTION["num_seeds"])

        seeds_csv = analysis_dir / f"seeds_round{rn}.csv"
        seeds.to_csv(seeds_csv, index=False)
        self._last_seeds_df = seeds  # 保存供下一轮命名使用
        logger.info(f"  Selected {len(seeds)} seeds")

        # ── 提取最佳分数 ──
        best_energy = None
        if self.anti_enabled:
            energy_col = "delta"
        else:
            energy_col = "target_energy"

        if energy_col in docking_results.columns:
            ok_mask = docking_results["docking_success"] == True
            if ok_mask.any():
                best_energy = float(docking_results.loc[ok_mask, energy_col].min())
                self.best_scores.append(best_energy)
                label = "Best delta" if self.anti_enabled else "Best binding energy"
                logger.info(f"  {label}: {best_energy:.2f} kcal/mol")

        # ── 构建摘要 ──
        summary = self._build_summary(
            rn, len(compounds_df), len(filtered_df),
            len(valid_pdbqt), int(n_ok), int(len(docking_results) - n_ok),
            len(seeds), best_energy, "completed",
        )

        # 保存摘要
        with open(round_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.history.append(summary)

        # ── Step 6: 计算化学报告 ──
        report_dir = round_dir / "06_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[Step 6/6] Generating cheminformatics report...")

        try:
            from report_generator import generate_round_report
            generate_round_report(
                docking_results_df=docking_results,
                round_num=rn,
                output_dir=report_dir,
                summary=summary,
            )
        except Exception as e:
            logger.warning(f"Report generation failed (non-fatal): {e}")

        logger.info(f"Round {rn} completed.")
        return summary

    def _fail(self, rn: int, reason: str, detail: str = "") -> Dict:
        """构建失败摘要"""
        summary = self._build_summary(rn, 0, 0, 0, 0, 0, 0, None, "failed")
        summary["failure_reason"] = reason
        if detail:
            summary["failure_detail"] = detail
        self.history.append(summary)
        return summary

    def _build_summary(
        self,
        rn: int, gen: int, filt: int, pdbqt: int,
        dock_ok: int, dock_fail: int, seeds: int,
        best_e: Optional[float], status: str,
    ) -> Dict:
        return {
            "round": rn,
            "timestamp": datetime.now().isoformat(),
            "generated": gen,
            "after_filter": filt,
            "pdbqt_prepared": pdbqt,
            "docking_success": dock_ok,
            "docking_failed": dock_fail,
            "seeds_selected": seeds,
            "best_binding_energy": best_e,
            "status": status,
        }

    # ============================================================
    # 迭代循环
    # ============================================================

    def run_iterative_screening(
        self,
        initial_seeds: List[str],
        max_rounds: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        运行完整的迭代筛选循环

        Args:
            initial_seeds: 初始种子 SMILES 列表
            max_rounds: 最大轮次

        Returns:
            所有轮次的摘要列表
        """
        max_rounds = max_rounds or ITERATION_CONFIG["max_rounds"]
        conv_rounds = ITERATION_CONFIG["convergence_rounds"]
        conv_threshold = ITERATION_CONFIG["convergence_threshold"]
        early_stop = ITERATION_CONFIG["early_stopping"]

        logger.info(f"\n{'='*60}")
        logger.info(f"  Iterative Screening: max {max_rounds} rounds")
        logger.info(f"  Convergence: {conv_rounds} rounds, {conv_threshold} kcal/mol")
        logger.info(f"{'='*60}")

        current_seeds = initial_seeds
        all_summaries = []

        for rn in range(1, max_rounds + 1):
            summary = self.run_round(seed_smiles=current_seeds, round_num=rn)
            all_summaries.append(summary)

            if summary.get("status") == "failed":
                logger.warning(f"Round {rn} failed, stopping.")
                if not ITERATION_CONFIG.get("continue_on_failure", False):
                    break

            # 提取下一轮种子
            seeds_csv = (
                DIRS["rounds"] / f"round_{rn:03d}"
                / "05_analysis" / f"seeds_round{rn}.csv"
            )
            if seeds_csv.exists():
                try:
                    seeds_df = pd.read_csv(seeds_csv)
                    if "SMILES" in seeds_df.columns:
                        current_seeds = seeds_df["SMILES"].dropna().tolist()
                        self._last_seeds_df = seeds_df
                        logger.info(
                            f"  Next seeds: {len(current_seeds)}"
                            + (f" (e.g. {seeds_df['Compound'].iloc[0]})"
                               if 'Compound' in seeds_df.columns else "")
                        )
                        if len(current_seeds) == 0:
                            logger.warning("No seeds selected — stopping iteration.")
                            break
                except Exception as e:
                    logger.warning(f"Could not read seeds: {e}")

            # 收敛检测
            if early_stop and self._check_convergence(conv_rounds, conv_threshold):
                logger.info(f"Converged at round {rn}.")
                break

            # 保存检查点
            if ITERATION_CONFIG.get("save_checkpoints", True):
                self._save_checkpoint(rn, summary)

        # 生成最终报告
        self._generate_final_report(all_summaries)
        return all_summaries

    def _check_convergence(
        self,
        rounds: int,
        threshold: float,
    ) -> bool:
        """检查是否收敛"""
        if len(self.best_scores) < rounds:
            return False

        recent = self.best_scores[-rounds:]
        for i in range(1, len(recent)):
            improvement = recent[i - 1] - recent[i]
            if improvement > threshold:
                return False
        return True

    def _save_checkpoint(self, rn: int, summary: Dict):
        """保存检查点"""
        ckpt_dir = DIRS["checkpoints"]
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "round": rn,
            "summary": summary,
            "history": self.history,
            "best_scores": self.best_scores,
            "timestamp": datetime.now().isoformat(),
        }
        with open(ckpt_dir / f"checkpoint_round{rn}.json", "w") as f:
            json.dump(ckpt, f, indent=2, default=str)

    def _generate_final_report(self, summaries: List[Dict]):
        """生成最终 JSON 报告"""
        report = {
            "pipeline": "Virtual Screening with AutoDock-GPU + Reinvent4",
            "total_rounds": len(summaries),
            "completed": sum(1 for s in summaries if s.get("status") == "completed"),
            "failed": sum(1 for s in summaries if s.get("status") == "failed"),
            "best_score_overall": (
                min(self.best_scores) if self.best_scores else None
            ),
            "rounds": summaries,
            "timestamp": datetime.now().isoformat(),
        }

        report_path = DIRS["output"] / "pipeline_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Final report saved: {report_path}")

    # ============================================================
    # 检查点恢复
    # ============================================================

    def load_checkpoint(self, round_num: int) -> Optional[Dict]:
        """
        加载指定轮次的检查点

        Args:
            round_num: 轮次编号

        Returns:
            检查点字典，或 None
        """
        ckpt_path = DIRS["checkpoints"] / f"checkpoint_round{round_num}.json"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            return None

        with open(ckpt_path) as f:
            ckpt = json.load(f)

        self.current_round = ckpt["round"]
        self.history = ckpt.get("history", [])
        self.best_scores = ckpt.get("best_scores", [])

        logger.info(f"Loaded checkpoint round {round_num}")
        return ckpt


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    orch = PipelineOrchestrator(
        receptor_pdbqt="output/prepared_protein/4dkl.pdbqt",
        center_x=10.0, center_y=25.0, center_z=12.0,
        size_x=30, size_y=30, size_z=30,
    )

    test_seeds = [
        "COC1=CC2=C(C=C1)C(=O)C(=CO2)C3=CC=C(C=C3)O",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC=N4",
    ]

    summary = orch.run_round(seed_smiles=test_seeds)
    print(json.dumps(summary, indent=2, default=str))

#!/usr/bin/env python3
"""
虚拟筛选迭代循环 — 主入口
Virtual Screening Iterative Loop — Main Entry Point

基于 AutoDock-GPU + Reinvent4 的分子定向进化全流程自动化。

用法示例:
    # 1. 交互式蛋白准备 + 对接
    python main.py --pdb-id 4dkl --interactive --max-rounds 5

    # 2. 使用本地蛋白文件 + JSON 配置
    python main.py --pdb output/prepared_protein/4dkl.pdbqt \
                   --center_x 10 --center_y 25 --center_z 12 \
                   --size_x 30 --size_y 30 --size_z 30

    # 3. 单轮测试模式
    python main.py --pdb output/prepared_protein/4dkl.pdbqt --test

    # 4. 查看 Conda 环境状态
    python main.py --status

    # 5. 从配置文件运行
    python main.py --config configs/example_config.json

    # 6. 使用种子文件
    python main.py --pdb-id 4dkl --seeds my_seeds.smi --max-rounds 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DIRS, ADGPU_CONFIG, REINVENT_CONFIG,
    SEED_SELECTION, ITERATION_CONFIG,
    create_directories, validate_config,
)

# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DIRS["logs"] / "screening_pipeline.log"),
    ],
)
logger = logging.getLogger("main")


# ============================================================
# CLI 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="🔄 Virtual Screening Pipeline — AutoDock-GPU + Reinvent4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式（推荐新手）
  python main.py --pdb-id 4dkl --interactive

  # 批量迭代模式
  python main.py --pdb output/prepared_protein/4dkl.pdbqt \\
                 --center_x 10 --center_y 25 --center_z 12 \\
                 --size_x 30 --size_y 30 --size_z 30 \\
                 --seeds seeds.smi --max-rounds 5

  # 使用配置文件
  python main.py --config configs/example_config.json

  # 查看环境状态
  python main.py --status
        """,
    )

    # ── 蛋白输入 ──
    parser.add_argument(
        "--pdb-id", type=str, default=None,
        help="主靶点 PDB 编号（如 4dkl），自动下载并准备蛋白",
    )
    parser.add_argument(
        "--anti-pdb-id", type=str, default=None,
        help="负靶点（anti-target）PDB 编号，用于选择性筛选",
    )
    parser.add_argument(
        "--pdb", "--receptor", type=str, default=None,
        help="本地主靶点受体 PDBQT 文件路径",
    )
    parser.add_argument(
        "--anti-pdb", type=str, default=None,
        help="本地负靶点受体 PDBQT 文件路径",
    )

    # ── 对接盒参数 ──
    parser.add_argument("--center_x", type=float, default=None)
    parser.add_argument("--center_y", type=float, default=None)
    parser.add_argument("--center_z", type=float, default=None)
    parser.add_argument("--size_x", type=int, default=None)
    parser.add_argument("--size_y", type=int, default=None)
    parser.add_argument("--size_z", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None,
                       help="GPU 设备编号（默认 0）")

    # ── 运行模式 ──
    parser.add_argument("--config", type=str, default=None,
                       help="JSON 配置文件路径")
    parser.add_argument("--test", action="store_true",
                       help="单轮测试模式（只运行一轮）")
    parser.add_argument("--interactive", action="store_true",
                       help="交互式口袋选择模式")
    parser.add_argument("--status", action="store_true",
                       help="显示 Conda 环境和工具状态后退出")
    parser.add_argument("--export-config", type=str, default=None,
                       help="导出当前配置到指定 JSON 文件后退出")

    # ── 迭代参数 ──
    parser.add_argument("--max-rounds", type=int, default=None,
                       help="最大迭代轮数（默认从配置读取）")
    parser.add_argument("--seeds", type=str, default=None,
                       help="初始种子 SMILES 文件路径（每行一个 SMILES）")
    parser.add_argument("--num-seeds", type=int, default=None,
                       help="每轮挑选的种子数量")

    return parser.parse_args()


# ============================================================
# 辅助函数
# ============================================================

def load_seeds_file(path: str) -> List[str]:
    """从文件加载种子 SMILES"""
    with open(path, "r") as f:
        seeds = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(seeds)} seed SMILES from {path}")
    return seeds


def load_json_config(path: str) -> dict:
    """从 JSON 文件加载配置"""
    with open(path, "r") as f:
        return json.load(f)


def print_banner(title: str):
    """打印格式化横幅"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# 主函数
# ============================================================

def main():
    """主入口"""
    args = parse_args()

    create_directories()

    # ── 特殊模式: 显示状态 ──
    if args.status:
        from conda_manager import CondaManager
        cm = CondaManager()
        cm.scan_environments()
        cm.assign_tools()
        cm.print_status_report()

        # 也显示配置验证
        warnings = validate_config()
        if warnings:
            print("\n⚠️  Configuration Warnings:")
            for w in warnings:
                print(f"  - {w}")
        return

    # ── 特殊模式: 导出配置 ──
    if args.export_config:
        from config import export_config
        path = export_config(Path(args.export_config))
        print(f"Config exported to: {path}")
        return

    # ── 验证配置 ──
    warnings = validate_config()
    for w in warnings:
        logger.warning(w)

    # ── 确定对接参数 ──
    receptor_pdbqt = None
    center_x = ADGPU_CONFIG["center_x"]
    center_y = ADGPU_CONFIG["center_y"]
    center_z = ADGPU_CONFIG["center_z"]
    size_x = ADGPU_CONFIG["size_x"]
    size_y = ADGPU_CONFIG["size_y"]
    size_z = ADGPU_CONFIG["size_z"]
    gpu = ADGPU_CONFIG.get("gpu", 0)

    # 从 JSON 配置文件加载
    if args.config:
        logger.info(f"Loading config: {args.config}")
        cfg = load_json_config(args.config)

        d = cfg.get("docking", {})
        receptor_pdbqt = d.get("receptor", receptor_pdbqt)
        center_x = d.get("center_x", center_x)
        center_y = d.get("center_y", center_y)
        center_z = d.get("center_z", center_z)
        size_x = d.get("size_x", size_x)
        size_y = d.get("size_y", size_y)
        size_z = d.get("size_z", size_z)
        gpu = d.get("gpu", gpu)

        # 也更新其他模块配置
        if "seed_selection" in cfg:
            SEED_SELECTION.update(cfg["seed_selection"])
        if "iteration" in cfg:
            ITERATION_CONFIG.update(cfg["iteration"])
        if "reinvent" in cfg:
            REINVENT_CONFIG.update(cfg["reinvent"])

    # 命令行参数覆盖
    if args.center_x is not None: center_x = args.center_x
    if args.center_y is not None: center_y = args.center_y
    if args.center_z is not None: center_z = args.center_z
    if args.size_x is not None: size_x = args.size_x
    if args.size_y is not None: size_y = args.size_y
    if args.size_z is not None: size_z = args.size_z
    if args.gpu is not None: gpu = args.gpu
    if args.num_seeds is not None: SEED_SELECTION["num_seeds"] = args.num_seeds

    # ── 蛋白准备（主靶点 + 负靶点）──
    anti_receptor_pdbqt = None
    anti_grid_params = None  # (cx, cy, cz, sx, sy, sz) for anti-target

    def _prepare_protein(pdb_id, interactive):
        """准备单个蛋白：下载 + 口袋检测 + PDBQT 转换"""
        from protein_preparator import ProteinPreparator
        prep = ProteinPreparator()
        if interactive:
            result = prep.prepare_protein_interactive(pdb_id)
        else:
            result = prep.prepare_protein_pipeline(pdb_id)
        return result, prep

    if args.pdb_id:
        print_banner(f"Target Protein Preparation: {args.pdb_id.upper()}")
        result, prep = _prepare_protein(args.pdb_id, args.interactive)

        if result["pdbqt_path"] and result["selected_pocket"]:
            receptor_pdbqt = str(result["pdbqt_path"])
            pocket = result["selected_pocket"]
            center_x = float(pocket["center"][0])
            center_y = float(pocket["center"][1])
            center_z = float(pocket["center"][2])
            spacing = ADGPU_CONFIG.get("grid_spacing", 0.375)
            size_x = max(20, int(pocket["size"][0] / spacing))
            size_y = max(20, int(pocket["size"][1] / spacing))
            size_z = max(20, int(pocket["size"][2] / spacing))
            logger.info(f"Target pocket: {pocket['name']}")
            logger.info(f"Target grid: center=({center_x:.2f},{center_y:.2f},{center_z:.2f}) size=({size_x},{size_y},{size_z})")
        elif result["pdbqt_path"]:
            receptor_pdbqt = str(result["pdbqt_path"])
            logger.warning("No pocket detected for target. Using default grid params.")
        else:
            logger.error("Target protein preparation failed.")
            sys.exit(1)

    elif args.pdb:
        receptor_pdbqt = args.pdb

    # ── 负靶点（anti-target）准备 ──
    if args.anti_pdb_id:
        print_banner(f"Anti-Target Protein Preparation: {args.anti_pdb_id.upper()}")
        anti_result, _ = _prepare_protein(args.anti_pdb_id, args.interactive)

        if anti_result["pdbqt_path"] and anti_result["selected_pocket"]:
            anti_receptor_pdbqt = str(anti_result["pdbqt_path"])
            anti_pocket = anti_result["selected_pocket"]
            anti_cx = float(anti_pocket["center"][0])
            anti_cy = float(anti_pocket["center"][1])
            anti_cz = float(anti_pocket["center"][2])
            spacing = ADGPU_CONFIG.get("grid_spacing", 0.375)
            anti_sx = max(20, int(anti_pocket["size"][0] / spacing))
            anti_sy = max(20, int(anti_pocket["size"][1] / spacing))
            anti_sz = max(20, int(anti_pocket["size"][2] / spacing))
            anti_grid_params = (anti_cx, anti_cy, anti_cz, anti_sx, anti_sy, anti_sz)
            logger.info(f"Anti-target pocket: {anti_pocket['name']}")
        elif anti_result["pdbqt_path"]:
            anti_receptor_pdbqt = str(anti_result["pdbqt_path"])
            anti_grid_params = (center_x, center_y, center_z, size_x, size_y, size_z)
            logger.warning("No pocket for anti-target. Using target grid params as fallback.")
        else:
            logger.error("Anti-target protein preparation failed.")
            sys.exit(1)

    elif args.anti_pdb:
        anti_receptor_pdbqt = args.anti_pdb
        anti_grid_params = (center_x, center_y, center_z, size_x, size_y, size_z)

    # 无参数运行 → 显示帮助
    if not receptor_pdbqt and not args.config and not args.status and not args.export_config:
        print("=" * 60)
        print("  Molecular Directed Evolution Pipeline")
        print("  AutoDock-GPU + REINVENT4 Mol2Mol")
        print("=" * 60)
        print()
        print("  Quick start:")
        print("    python main.py --pdb-id 4dkl --test")
        print("    python main.py --pdb-id 4dkl --anti-pdb-id 7eu8 --max-rounds 5")
        print("    python main.py --status")
        print()
        print("  Run 'python main.py --help' for all options.")
        print("=" * 60)
        sys.exit(0)

    # 验证受体
    if not receptor_pdbqt:
        logger.error(
            "No receptor specified! Use --pdb-id to download, "
            "--pdb for a local file, or --config for a config file."
        )
        sys.exit(1)

    if not Path(receptor_pdbqt).exists():
        logger.error(f"Receptor file not found: {receptor_pdbqt}")
        sys.exit(1)

    # ── 加载种子 ──
    if args.seeds:
        initial_seeds = load_seeds_file(args.seeds)
    else:
        # 默认种子（用于测试）
        initial_seeds = [
            "COC1=CC2=C(C=C1)C(=O)C(=CO2)C3=CC=C(C=C3)O",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC=N4",
        ]
        logger.info(f"Using {len(initial_seeds)} default seed SMILES")

    # ── 创建编排器 ──
    from pipeline_orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator(
        receptor_pdbqt=receptor_pdbqt,
        center_x=center_x, center_y=center_y, center_z=center_z,
        size_x=size_x, size_y=size_y, size_z=size_z,
        gpu=gpu,
        anti_receptor_pdbqt=anti_receptor_pdbqt,
        anti_grid_params=anti_grid_params,
    )

    # ── 运行 ──
    if args.test:
        print_banner("Single Round Test Mode")
        print(f"  Target receptor:    {receptor_pdbqt}")
        print(f"  Target grid:        center=({center_x:.1f},{center_y:.1f},{center_z:.1f}) size=({size_x},{size_y},{size_z})")
        if anti_receptor_pdbqt:
            print(f"  Anti-target receptor: {anti_receptor_pdbqt}")
        print(f"  GPU: {gpu}")
        print(f"{'='*60}")

        summary = orch.run_round(seed_smiles=initial_seeds)
        print(f"\n✅ Test complete!")
        print(json.dumps(summary, indent=2, default=str))
    else:
        max_rounds = args.max_rounds or ITERATION_CONFIG["max_rounds"]

        print_banner(f"Iterative Screening — Max {max_rounds} Rounds")
        print(f"  Target receptor:    {receptor_pdbqt}")
        if anti_receptor_pdbqt:
            print(f"  Anti-target receptor: {anti_receptor_pdbqt}")
        print(f"  Grid center: ({center_x:.1f}, {center_y:.1f}, {center_z:.1f})")
        print(f"  Grid size: ({size_x}, {size_y}, {size_z})")
        print(f"  GPU: {gpu}")
        print(f"  Seeds per round: {SEED_SELECTION['num_seeds']}")
        print(f"  Seed strategy: {SEED_SELECTION['strategy']}")
        print(f"{'='*60}")

        summaries = orch.run_iterative_screening(
            initial_seeds=initial_seeds,
            max_rounds=max_rounds,
        )

        print(f"\n{'='*60}")
        print(f"  🎉 Pipeline Complete!")
        print(f"  Total rounds: {len(summaries)}")
        for s in summaries:
            status = s.get("status", "?")
            energy = s.get("best_binding_energy", "N/A")
            icon = "✅" if status == "completed" else "❌"
            print(f"  {icon} Round {s['round']}: {status} "
                  f"(best: {energy:.2f} kcal/mol)" if isinstance(energy, (int, float)) else
                  f"  {icon} Round {s['round']}: {status} (best: {energy})")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

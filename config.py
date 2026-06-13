#!/usr/bin/env python3
"""
虚拟筛选迭代循环 — 全局配置文件
Virtual Screening Iterative Loop — Global Configuration

所有用户可调参数集中于此文件。
支持通过 JSON 配置文件或环境变量覆盖。

用法:
    from config import (
        DIRS, ADGPU_CONFIG, REINVENT_CONFIG, PROPERTY_FILTERS,
        SEED_SELECTION, ITERATION_CONFIG, CONDA_TOOLS,
        create_directories, validate_config, load_user_config,
    )
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# ============================================================
# 项目根目录
# ============================================================
PROJECT_ROOT = Path(__file__).parent.absolute()


# ============================================================
# 目录结构配置
# ============================================================
DIRS: Dict[str, Path] = {
    # 输入数据
    "data": PROJECT_ROOT / "data",
    # 日志
    "logs": PROJECT_ROOT / "logs",
    # 配置文件
    "configs": PROJECT_ROOT / "configs",
    # 总输出
    "output": PROJECT_ROOT / "output",
    # 临时文件
    "temp": PROJECT_ROOT / "temp",
}

# 输出子目录
DIRS["raw_pdb"] = DIRS["output"] / "raw_pdb"
DIRS["prepared_protein"] = DIRS["output"] / "prepared_protein"
DIRS["grid"] = DIRS["output"] / "grid"
DIRS["rounds"] = DIRS["output"] / "rounds"
DIRS["checkpoints"] = DIRS["output"] / "checkpoints"
DIRS["tool_compounds"] = DIRS["output"] / "tool_compounds"


# ============================================================
# Conda 环境配置 — 工具 → 环境的映射
# ============================================================
# 策略:
#   1. 如果设置了环境变量 CONDA_ENV_<TOOL>，优先使用
#   2. 否则使用 conda_manager.py 的自动扫描结果
#   3. 最后回退到默认值

CONDA_TOOLS: Dict[str, str] = {
    # 核心化学信息学工具
    "rdkit": os.environ.get("CONDA_ENV_RDKIT", "reinvent4"),
    # Reinvent4 分子生成
    "reinvent4": os.environ.get("CONDA_ENV_REINVENT4", "reinvent4"),
    # Meeko — PDBQT 配体准备
    "meeko": os.environ.get("CONDA_ENV_MEEKO", "reinvent4"),
    # PDBFixer / OpenMM — 蛋白结构修复
    "pdbfixer": os.environ.get("CONDA_ENV_PDBFIXER", "reinvent4"),
    # 网络请求（PDB 下载、ChEMBL 查询）
    "requests": os.environ.get("CONDA_ENV_REQUESTS", "reinvent4"),
    # ChEMBL Web Resource Client
    "chembl": os.environ.get("CONDA_ENV_CHEMBL", "reinvent4"),
    # AutoDock-GPU（独立二进制，通常不需要 conda 环境）
    "autodock_gpu": os.environ.get("CONDA_ENV_ADGPU", ""),
}

# 方便旧代码兼容
CONDA_ENVS = {
    "reinvent4": CONDA_TOOLS["reinvent4"],
}


# ============================================================
# 外部工具自动发现
# ============================================================

def _find_binary(name: str, fallback: str = "") -> str:
    """在 PATH 和 conda 环境中查找可执行文件"""
    import shutil as _sh
    # 1. 环境变量
    env_key = name.upper() + "_BIN"
    if os.environ.get(env_key):
        return os.environ[env_key]
    # 2. 直接 which
    found = _sh.which(name)
    if found:
        return found
    # 3. 常见安装位置
    for base in [Path.home() / "miniconda3", Path.home() / "anaconda3",
                  Path("/opt/conda"), Path("/usr/local")]:
        for sub in ["bin", "envs"]:
            p = base / sub / name
            if p.exists():
                return str(p)
    # 4. 搜索所有 conda 环境
    for env_dir in [Path.home() / "miniconda3" / "envs",
                     Path.home() / "anaconda3" / "envs"]:
        if env_dir.exists():
            for env_path in env_dir.iterdir():
                p = env_path / "bin" / name
                if p.exists():
                    return str(p)
    return fallback or name

TOOLS: Dict[str, str] = {
    "reinvent4": _find_binary("reinvent", "reinvent"),
    "autodock_gpu": _find_binary("autodock_gpu_128wi",
        str(Path.home() / "bio_tools/AutoDock-GPU/bin/autodock_gpu_128wi")),
    "prepare_ligand4": _find_binary("prepare_ligand4"),
    "prepare_gpf4": _find_binary("prepare_gpf4"),
    "autogrid4": _find_binary("autogrid4", "/usr/bin/autogrid4"),
    "obabel": _find_binary("obabel", "obabel"),
}

# Reinvent4 Mol2Mol prior 模型
MOL2MOL_PRIOR_PATH = os.environ.get(
    "MOL2MOL_PRIOR",
    str(Path.home() / "REINVENT4-main/prior/mol2mol_medium_similarity.prior"),
)
if not Path(MOL2MOL_PRIOR_PATH).exists():
    # 搜索
    for p in Path.home().rglob("mol2mol_medium_similarity.prior"):
        MOL2MOL_PRIOR_PATH = str(p)
        break


# ============================================================
# Reinvent4 配置
# ============================================================
REINVENT_CONFIG: Dict[str, Any] = {
    # Reinvent4 Mol2Mol 预训练模型路径
    "prior_model": "/home/xiayun-huang/REINVENT4-main/prior/mol2mol_medium_similarity.prior",
    # Reinvent4 可执行文件
    "reinvent_bin": "/home/xiayun-huang/miniconda3/envs/reinvent4/bin/reinvent",
    # 每轮生成的化合物总数
    "num_smiles": 5000,
    # 每个种子生成的分子数（自动 = num_smiles / num_seeds）
    "num_per_seed": None,
    # Mol2Mol 采样温度 (0.5-1.0, 越低越保守)
    "temperature": 0.8,
    # 采样策略: "multinomial" | "beamsearch"
    "sample_strategy": "multinomial",
}


# ============================================================
# 理化性质筛选配置 (Lipinski Rule-of-5 及扩展)
# ============================================================
PROPERTY_FILTERS: Dict[str, Any] = {
    # ── 库生成阶段：宽松 Lipinski（保留化学多样性）──
    "molecular_weight": {"min": 150, "max": 500},
    "logp": {"min": -2.0, "max": 5.0},
    "hbd": {"max": 5},
    "hba": {"max": 10},
    "rotatable_bonds": {"max": 10},
    "tpsa": {"min": 0, "max": 140},
    "ring_count": {"max": 6},
    # CNS 严格参数在种子选择阶段单独施加，此处关闭
    "fsp3": None,
    "remove_strong_acids": False,
    "remove_pains": True,
    "remove_brenk": False,
    "require_lipinski": True,
}


# ============================================================
# CNS 药物专属性质过滤（血脑屏障穿透性）
# ============================================================
CNS_FILTER: Dict[str, Any] = {
    # 是否启用 CNS 过滤
    "enabled": False,
    # 分子量上限
    "mw_max": 450,
    # LogP 范围
    "logp_min": 1.0,
    "logp_max": 4.5,
    # TPSA 上限
    "tpsa_max": 90,
    # HBD 上限
    "hbd_max": 2,
    # HBA 上限
    "hba_max": 7,
}


# ============================================================
# AutoDock-GPU 对接配置
# ============================================================
ADGPU_CONFIG: Dict[str, Any] = {
    # ── 对接盒参数 ──
    # 对接盒中心坐标 (Å)
    "center_x": 0.0,
    "center_y": 0.0,
    "center_z": 0.0,
    # 对接盒网格点数 (每个方向)
    "size_x": 30,
    "size_y": 30,
    "size_z": 30,
    # 网格间距 (Å)，AutoDock 默认 0.375
    "grid_spacing": 0.375,

    # ── 对接参数 ──
    # 输出的对接构象数量
    "num_poses": 20,
    # 每个配体的对接运行次数
    "num_runs": 10,
    # 种群大小 (GA 参数)
    "population_size": 200,
    # 最大评估次数
    "max_evals": 5_000_000,
    # 最大代数
    "max_generations": 50_000,

    # ── GPU 配置 ──
    # GPU 设备编号
    "gpu": 0,

    # ── 文件 ──
    # 受体 PDBQT 文件路径（通过命令行设置或交互式选择）
    "receptor": "",
}


# ============================================================
# 种子筛选配置
# ============================================================
SEED_SELECTION: Dict[str, Any] = {
    # 每轮挑选的种子数量
    "num_seeds": 50,
    # 对接分数阈值 (kcal/mol)，低于此值的视为"好"
    "binding_energy_threshold": -7.0,
    # 挑选策略: "top_n" | "threshold" | "cluster" | "cns_cluster"
    # 推荐: 双靶点用 "cns_cluster"，单靶点用 "top_n"
    "strategy": "cns_cluster",
    # 聚类挑选时的 Tanimoto 相似度阈值
    "cluster_similarity": 0.7,
    # 是否启用 CNS 过滤
    "enable_cns_filter": False,
    # 是否启用 PAINS 过滤
    "enable_pains_filter": True,
    # 最大种子数（安全上限）
    "max_seeds": 200,
    # 双靶点筛选: delta 阈值（低于此值视为"选择性好"）
    "delta_threshold": -0.5,
    # 主靶点能量阈值 (kcal/mol)
    "target_energy_threshold": -7.0,
    # 多样性控制
    "diversity_similarity_start": 0.4,
    "diversity_similarity_max": 0.7,
    # CNS 严格物性筛选（血脑屏障穿透优化）
    "cns_strict": {
        "MW_max": 360,
        "cLogP_min": 2.0, "cLogP_max": 4.0,
        "TPSA_max": 70,
        "HBD_max": 1,
        "HBA_max": 7,
        "RotBonds_max": 5,
        "Fsp3_min": 0.45,
        "no_strong_acids": True,
    },
}


# ============================================================
# 迭代循环配置
# ============================================================
ITERATION_CONFIG: Dict[str, Any] = {
    # 最大迭代轮数
    "max_rounds": 10,
    # 收敛（选择性进化需要充分迭代，暂关早停）
    "convergence_rounds": 10,
    "convergence_threshold": 0.1,
    "early_stopping": False,
    # 是否保存每轮检查点
    "save_checkpoints": True,
    # 是否在失败时继续下一轮（而不是终止）
    "continue_on_failure": False,
}


# ============================================================
# 日志配置
# ============================================================
LOGGING: Dict[str, Any] = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(DIRS["logs"] / "screening_pipeline.log"),
    "console": True,
}


# ============================================================
# 辅助函数
# ============================================================

def create_directories() -> None:
    """创建所有必要的目录"""
    for name, path in DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
    logger.info(f"All directories created under {PROJECT_ROOT}")


def validate_config() -> List[str]:
    """
    验证配置是否完整，返回警告列表

    Returns:
        警告字符串列表
    """
    warnings = []

    # 检查 AutoDock-GPU
    adgpu_path = TOOLS.get("autodock_gpu", "autodock-gpu")
    if "/" in adgpu_path:
        if not Path(adgpu_path).exists() and not shutil.which(adgpu_path):
            warnings.append(
                f"AutoDock-GPU not found at '{adgpu_path}'. "
                f"Set TOOLS['autodock_gpu'] in config or ADGPU_BIN env var."
            )
    else:
        if not shutil.which(adgpu_path):
            warnings.append(
                f"AutoDock-GPU ('{adgpu_path}') not in PATH. "
                f"Install AutoDock-GPU or set ADGPU_BIN env var."
            )

    # 检查受体文件（如果已设置）
    receptor = ADGPU_CONFIG.get("receptor", "")
    if receptor and not Path(receptor).exists():
        warnings.append(f"Receptor file not found: {receptor}")

    # 检查 prepare_receptor4
    prep4 = TOOLS.get("prepare_receptor4", "")
    if prep4 and "/" in prep4 and not Path(prep4).exists():
        warnings.append(
            f"prepare_receptor4 not found at '{prep4}'. "
            f"PDB→PDBQT conversion may fail."
        )

    # 检查 Reinvent4
    reinvent_path = TOOLS.get("reinvent4", "reinvent")
    if "/" in reinvent_path:
        if not Path(reinvent_path).exists():
            warnings.append(f"Reinvent4 not found at '{reinvent_path}'.")
    else:
        if not shutil.which(reinvent_path):
            warnings.append(
                f"Reinvent4 ('{reinvent_path}') not in PATH. "
                f"Falling back to mock data mode."
            )

    return warnings


def load_user_config(config_path: str) -> Dict[str, Any]:
    """
    从 JSON 文件加载用户配置

    配置文件的任何字段会覆盖默认配置中对应的值。
    支持嵌套覆盖（如 {"docking": {"center_x": 5.0}} 只覆盖 center_x）。

    Args:
        config_path: JSON 配置文件路径

    Returns:
        完整配置字典
    """
    with open(config_path, "r") as f:
        user_cfg = json.load(f)

    # 深度合并配置
    merged = {
        "reinvent": {**REINVENT_CONFIG},
        "property_filters": {**PROPERTY_FILTERS},
        "docking": {**ADGPU_CONFIG},
        "seed_selection": {**SEED_SELECTION},
        "iteration": {**ITERATION_CONFIG},
        "cns_filter": {**CNS_FILTER},
    }

    for section in merged:
        if section in user_cfg:
            merged[section].update(user_cfg[section])

    # 也支持顶层字段
    for key in ["reinvent", "property_filters", "docking",
                 "seed_selection", "iteration"]:
        mapped_key = {
            "reinvent": "reinvent",
            "property_filters": "property_filters",
            "docking": "docking",
            "seed_selection": "seed_selection",
            "iteration": "iteration",
        }.get(key, key)
        if key in user_cfg:
            merged[mapped_key].update(user_cfg[key])

    logger.info(f"Loaded user config from: {config_path}")
    return merged


def export_config(output_path: Optional[Path] = None) -> Path:
    """
    导出当前配置为 JSON 文件（用于备份/分享）

    Args:
        output_path: 输出路径，默认保存到 configs/

    Returns:
        输出文件路径
    """
    if output_path is None:
        output_path = DIRS["configs"] / "current_config.json"

    config_dict = {
        "reinvent": REINVENT_CONFIG,
        "property_filters": PROPERTY_FILTERS,
        "cns_filter": CNS_FILTER,
        "docking": {k: v for k, v in ADGPU_CONFIG.items()
                     if k != "receptor"},
        "seed_selection": SEED_SELECTION,
        "iteration": ITERATION_CONFIG,
    }

    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    logger.info(f"Config exported to: {output_path}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    create_directories()
    print("Directories created.")

    warnings = validate_config()
    if warnings:
        print("\n⚠️  Configuration Warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n✅ Configuration is valid.")

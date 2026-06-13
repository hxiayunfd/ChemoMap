"""
虚拟筛选迭代循环流水线 — Virtual Screening Iterative Loop Pipeline

基于 AutoDock-GPU + Reinvent4 的分子定向进化全流程自动化框架。

模块:
    conda_manager      — Conda 环境自动发现与管理
    config             — 全局配置系统
    protein_preparator — PDB 获取、修复、口袋检测、PDBQT 转换
    ligand_preparator  — SMILES → 3D SDF → PDBQT 配体准备
    docking_engine     — AutoDock-GPU 分子对接
    property_filter    — 理化性质筛选 (Lipinski, PAINS 等)
    seed_selector      — 对接结果分析与种子挑选
    reinvent_generator — Reinvent4 新分子生成
    pipeline_orchestrator — 流水线编排与迭代控制
    post_screening_clustering — 筛选后聚类分析 (CNS/PAINS/Tanimoto)
    tool_compound_generator  — ChEMBL 工具化合物获取

用法:
    # 命令行
    python main.py --pdb-id 4dkl --max-rounds 5

    # Python API
    from pipeline_orchestrator import PipelineOrchestrator
    orch = PipelineOrchestrator(receptor_pdbqt="protein.pdbqt", ...)
    orch.run_iterative_screening(initial_seeds=["CCO", ...], max_rounds=5)
"""

__version__ = "2.0.0"
__author__ = "Molecular Directed Evolution Pipeline Team"
__description__ = "AutoDock-GPU + Reinvent4 based virtual screening pipeline"

# 便捷导出
from .config import (
    DIRS, ADGPU_CONFIG, REINVENT_CONFIG,
    PROPERTY_FILTERS, SEED_SELECTION, ITERATION_CONFIG,
    TOOLS, CONDA_TOOLS,
    create_directories, validate_config, load_user_config,
)

# 条件导入：不强制要求所有依赖都存在
try:
    from .conda_manager import CondaManager, get_conda_manager
except ImportError:
    CondaManager = None
    get_conda_manager = None

try:
    from .protein_preparator import ProteinPreparator
except ImportError:
    ProteinPreparator = None

try:
    from .ligand_preparator import LigandPreparator
except ImportError:
    LigandPreparator = None

try:
    from .docking_engine import DockingEngine
except ImportError:
    DockingEngine = None

try:
    from .property_filter import PropertyFilter
except ImportError:
    PropertyFilter = None

try:
    from .seed_selector import SeedSelector
except ImportError:
    SeedSelector = None

try:
    from .reinvent_generator import ReinventGenerator
except ImportError:
    ReinventGenerator = None

try:
    from .pipeline_orchestrator import PipelineOrchestrator
except ImportError:
    PipelineOrchestrator = None

try:
    from .post_screening_clustering import PostScreeningClustering
except ImportError:
    PostScreeningClustering = None

try:
    from .tool_compound_generator import ToolCompoundGenerator
except ImportError:
    ToolCompoundGenerator = None

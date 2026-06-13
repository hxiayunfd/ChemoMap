#!/usr/bin/env python3
"""
模块: Conda 环境管理器
Module: Conda Environment Manager

自动扫描系统中的所有 Conda 环境，
为每个工具（RDKit、AutoDock-GPU、Reinvent4）匹配正确的 Conda 环境，
并通过子进程在正确的环境中执行代码。

功能:
1. 自动扫描所有 Conda 环境
2. 检测每个环境中可用的工具
3. 为各工具绑定正确的 conda env
4. 统一的子进程调用接口

用法:
    from conda_manager import CondaManager

    cm = CondaManager()
    cm.scan_environments()
    cm.assign_tools()

    # 在指定 conda 环境中执行 Python 脚本
    output = cm.run_python("reinvent4", "print('hello')")

    # 在指定 conda 环境中执行任意命令
    cm.run_command("adgpu", ["autodock-gpu", "--receptor", "rec.pdbqt", ...])
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# ============================================================
# 工具定义：每个工具需要哪些 Python 包 / 可执行文件
# ============================================================
TOOL_DEFINITIONS = {
    "rdkit": {
        "description": "RDKit — 化学信息学工具包",
        "python_packages": ["rdkit"],
        "executables": [],
        "default_env_name": "rdkit",
    },
    "reinvent4": {
        "description": "Reinvent4 — 分子从头生成与脚手架跳跃",
        "python_packages": ["reinvent"],
        "executables": ["reinvent"],
        "default_env_name": "reinvent4",
    },
    "autodock_gpu": {
        "description": "AutoDock-GPU — GPU 加速分子对接",
        "python_packages": [],
        "executables": ["autodock-gpu"],
        "default_env_name": "adgpu",
    },
    "meeko": {
        "description": "Meeko — 配体 PDBQT 准备工具",
        "python_packages": ["meeko"],
        "executables": [],
        "default_env_name": "reinvent4",
    },
    "pdbfixer": {
        "description": "PDBFixer / OpenMM — 蛋白结构修复",
        "python_packages": ["pdbfixer", "openmm"],
        "executables": [],
        "default_env_name": "reinvent4",
    },
    "requests": {
        "description": "requests — HTTP 请求库（PDB 下载、ChEMBL 查询）",
        "python_packages": ["requests"],
        "executables": [],
        "default_env_name": "reinvent4",
    },
    "chembl": {
        "description": "ChEMBL Web Resource Client",
        "python_packages": ["chembl_webresource_client"],
        "executables": [],
        "default_env_name": "reinvent4",
    },
}


class CondaManager:
    """
    Conda 环境管理器

    自动发现系统中的 Conda 环境，
    检测每个环境中安装的工具，
    并为各工具绑定正确的执行环境。

    Examples
    --------
    >>> cm = CondaManager()
    >>> cm.scan_environments()
    >>> cm.assign_tools()
    >>> print(cm.tool_env_map)
    {'rdkit': 'reinvent4', 'reinvent4': 'reinvent4', ...}
    >>> cm.run_python("rdkit", "from rdkit import Chem; print('OK')")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 Conda 管理器

        Args:
            config: 可选配置字典，可包含:
                - conda_bin: conda 可执行文件路径
                - env_overrides: {tool_name: env_name} 手动指定工具环境
                - auto_detect: 是否自动扫描环境
        """
        self.config = config or {}

        # 定位 conda 可执行文件
        self.conda_bin = self.config.get("conda_bin")
        if not self.conda_bin:
            self.conda_bin = self._find_conda()

        # 手动覆盖的环境映射
        self.env_overrides: Dict[str, str] = self.config.get("env_overrides", {})

        # 扫描结果
        self.available_envs: List[str] = []          # 所有可用的 conda 环境名
        self.env_packages: Dict[str, List[str]] = {}  # 每个环境中安装的 Python 包
        self.env_executables: Dict[str, List[str]] = {}  # 每个环境中可用的可执行文件

        # 工具→环境映射结果
        self.tool_env_map: Dict[str, str] = {}       # {tool_name: conda_env_name}
        self.tool_status: Dict[str, str] = {}         # {tool_name: "ok"|"missing"|"fallback"}

        # 是否已经扫描过
        self._scanned = False

    # ============================================================
    # Conda 定位
    # ============================================================

    def _find_conda(self) -> str:
        """
        在系统中定位 conda 可执行文件

        Returns:
            conda 可执行文件的路径
        """
        # 1. 环境变量
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe and os.path.exists(conda_exe):
            return conda_exe

        # 2. PATH 中查找
        found = shutil.which("conda")
        if found:
            return found

        # 3. 常见安装位置
        common_paths = [
            os.path.expanduser("~/miniconda3/bin/conda"),
            os.path.expanduser("~/anaconda3/bin/conda"),
            os.path.expanduser("~/conda/bin/conda"),
            "/opt/conda/bin/conda",
            "/usr/local/bin/conda",
        ]
        for p in common_paths:
            if os.path.exists(p):
                return p

        # 4. 最后的回退：尝试 mamba
        mamba = shutil.which("mamba")
        if mamba:
            logger.warning("conda not found, using mamba as fallback")
            return mamba

        logger.warning("Could not locate conda executable. "
                       "Please set CONDA_EXE environment variable or "
                       "configure conda_bin in config.")
        return "conda"  # 回退值

    # ============================================================
    # 环境扫描
    # ============================================================

    def scan_environments(self) -> List[str]:
        """
        扫描系统中所有的 Conda 环境

        运行 `conda env list` 获取所有环境。

        Returns:
            环境名称列表
        """
        if self._scanned:
            return self.available_envs

        logger.info(f"Scanning conda environments (conda: {self.conda_bin})...")

        try:
            result = subprocess.run(
                [self.conda_bin, "env", "list", "--json"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # conda env list --json 返回 {"envs": ["/path/to/env1", ...]}
                env_paths = data.get("envs", [])
                self.available_envs = [Path(p).name for p in env_paths]

                # 也记录完整路径
                self._env_paths = {Path(p).name: p for p in env_paths}

                logger.info(f"Found {len(self.available_envs)} conda environment(s): "
                           f"{', '.join(self.available_envs)}")
            else:
                logger.warning(f"conda env list failed: {result.stderr[:200]}")
                self.available_envs = ["base"]  # 至少回退到 base
        except Exception as e:
            logger.warning(f"Failed to scan conda environments: {e}")
            self.available_envs = ["base"]

        self._scanned = True
        return self.available_envs

    def detect_env_packages(self, env_name: str) -> List[str]:
        """
        检测指定 Conda 环境中安装的 Python 包

        运行 `conda run -n <env> pip list --format=json`

        Args:
            env_name: Conda 环境名称

        Returns:
            已安装的 Python 包名称列表（小写）
        """
        try:
            result = subprocess.run(
                [self.conda_bin, "run", "-n", env_name,
                 "python", "-c",
                 "import pkg_resources; "
                 "print([p.project_name.lower() for p in pkg_resources.working_set])"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                # 解析 Python 列表输出
                packages = eval(result.stdout.strip())
                return packages
        except Exception as e:
            logger.debug(f"Failed to detect packages in env '{env_name}': {e}")

        return []

    def detect_env_executables(self, env_name: str) -> List[str]:
        """
        检测指定 Conda 环境中可用的可执行文件

        Args:
            env_name: Conda 环境名称

        Returns:
            可执行文件名列表
        """
        try:
            result = subprocess.run(
                [self.conda_bin, "run", "-n", env_name,
                 "bash", "-c",
                 "compgen -c | sort -u | head -500"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except Exception as e:
            logger.debug(f"Failed to detect executables in env '{env_name}': {e}")

        return []

    # ============================================================
    # 工具→环境分配
    # ============================================================

    def assign_tools(self) -> Dict[str, str]:
        """
        为每个工具分配合适的 Conda 环境

        策略（按优先级）:
        1. 用户手动覆盖 (env_overrides)
        2. 扫描所有环境，选择第一个满足所有依赖的
        3. 使用默认环境名（如果存在）
        4. 回退到 base 环境

        Returns:
            {tool_name: conda_env_name} 映射
        """
        if not self._scanned:
            self.scan_environments()

        # 预检测所有环境（缓存）
        for env_name in self.available_envs:
            if env_name not in self.env_packages:
                self.env_packages[env_name] = self.detect_env_packages(env_name)
            if env_name not in self.env_executables:
                self.env_executables[env_name] = self.detect_env_executables(env_name)

        for tool_name, tool_def in TOOL_DEFINITIONS.items():
            # 优先级 1: 用户手动覆盖
            if tool_name in self.env_overrides:
                env_name = self.env_overrides[tool_name]
                if env_name in self.available_envs:
                    self.tool_env_map[tool_name] = env_name
                    self.tool_status[tool_name] = "ok"
                    continue

            # 优先级 2: 扫描匹配
            best_env = self._find_best_env(tool_def)
            if best_env:
                self.tool_env_map[tool_name] = best_env
                self.tool_status[tool_name] = "ok"
                continue

            # 优先级 3: 使用默认名
            default_env = tool_def.get("default_env_name", "base")
            if default_env in self.available_envs:
                self.tool_env_map[tool_name] = default_env
                self.tool_status[tool_name] = "fallback"
                continue

            # 优先级 4: 回退到 base
            if "base" in self.available_envs:
                self.tool_env_map[tool_name] = "base"
                self.tool_status[tool_name] = "fallback"
            else:
                # 使用第一个可用环境
                if self.available_envs:
                    self.tool_env_map[tool_name] = self.available_envs[0]
                    self.tool_status[tool_name] = "missing"
                else:
                    self.tool_env_map[tool_name] = "base"
                    self.tool_status[tool_name] = "missing"

        self._print_assignment_summary()
        return self.tool_env_map

    def _find_best_env(self, tool_def: Dict) -> Optional[str]:
        """
        查找满足所有依赖的最佳环境

        Args:
            tool_def: TOOL_DEFINITIONS 中的工具定义

        Returns:
            环境名称，或 None
        """
        for env_name in self.available_envs:
            packages = self.env_packages.get(env_name, [])
            executables = self.env_executables.get(env_name, [])

            # 检查 Python 包
            pkg_ok = all(
                pkg in packages
                for pkg in tool_def.get("python_packages", [])
            )

            # 检查可执行文件
            exe_ok = all(
                exe in executables
                for exe in tool_def.get("executables", [])
            )

            if pkg_ok and exe_ok:
                return env_name

        return None

    def _print_assignment_summary(self):
        """打印工具→环境分配摘要"""
        logger.info("=" * 60)
        logger.info("  Conda Environment Assignment Summary")
        logger.info("=" * 60)
        for tool_name in sorted(self.tool_env_map.keys()):
            env = self.tool_env_map[tool_name]
            status = self.tool_status.get(tool_name, "unknown")
            desc = TOOL_DEFINITIONS.get(tool_name, {}).get("description", tool_name)
            status_icon = {"ok": "✅", "fallback": "⚠️", "missing": "❌"}.get(status, "❓")
            logger.info(f"  {status_icon} {tool_name:20s} -> {env:15s}  ({desc})")
        logger.info("=" * 60)

    # ============================================================
    # 统一的子进程执行接口
    # ============================================================

    def run_python(
        self,
        tool_name: str,
        script: str,
        timeout: int = 300,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        在指定工具的 Conda 环境中执行 Python 脚本

        Args:
            tool_name: 工具名称（如 "rdkit", "reinvent4"）
            script: Python 脚本内容
            timeout: 超时时间（秒）
            capture_output: 是否捕获标准输出

        Returns:
            subprocess.CompletedProcess 对象

        Raises:
            RuntimeError: 脚本执行失败
        """
        env_name = self.get_env(tool_name)
        cmd = [
            self.conda_bin, "run", "-n", env_name,
            "python", "-c", script,
        ]
        logger.debug(f"[{tool_name}] Running in env '{env_name}'")

        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            stderr_snippet = (result.stderr or "")[:500]
            raise RuntimeError(
                f"Script failed in env '{env_name}' (tool={tool_name}): "
                f"{stderr_snippet}"
            )

        return result

    def run_command(
        self,
        tool_name: str,
        args: List[str],
        timeout: int = 600,
        cwd: Optional[Path] = None,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        在指定工具的 Conda 环境中执行任意命令

        Args:
            tool_name: 工具名称
            args: 命令参数列表
            timeout: 超时时间（秒）
            cwd: 工作目录
            capture_output: 是否捕获标准输出

        Returns:
            subprocess.CompletedProcess 对象
        """
        env_name = self.get_env(tool_name)
        cmd = [self.conda_bin, "run", "-n", env_name] + args
        logger.debug(f"[{tool_name}] Running: {' '.join(cmd)}")

        return subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )

    def run_command_direct(
        self,
        args: List[str],
        timeout: int = 600,
        cwd: Optional[Path] = None,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        直接执行命令（不使用 conda，用于独立可执行文件如 prepare_receptor4）

        Args:
            args: 命令参数列表
            timeout: 超时时间（秒）
            cwd: 工作目录
            capture_output: 是否捕获标准输出

        Returns:
            subprocess.CompletedProcess 对象
        """
        logger.debug(f"Running (direct): {' '.join(args)}")
        return subprocess.run(
            args,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )

    # ============================================================
    # 查询接口
    # ============================================================

    def get_env(self, tool_name: str) -> str:
        """
        获取工具对应的 Conda 环境名称

        Args:
            tool_name: 工具名称

        Returns:
            Conda 环境名称
        """
        if tool_name not in self.tool_env_map:
            self.assign_tools()

        env = self.tool_env_map.get(tool_name)
        if env is None:
            # 最终回退
            logger.warning(f"No conda environment assigned for '{tool_name}', "
                          f"using 'base'")
            return "base"
        return env

    def is_tool_available(self, tool_name: str) -> bool:
        """
        检查工具是否可用

        Args:
            tool_name: 工具名称

        Returns:
            是否可用
        """
        if tool_name not in self.tool_env_map:
            self.assign_tools()
        return self.tool_status.get(tool_name) != "missing"

    def get_status_report(self) -> Dict[str, Any]:
        """
        获取完整的状态报告

        Returns:
            包含所有环境信息的字典
        """
        if not self._scanned:
            self.scan_environments()
            self.assign_tools()

        return {
            "conda_binary": self.conda_bin,
            "available_environments": self.available_envs,
            "tool_assignments": dict(self.tool_env_map),
            "tool_status": dict(self.tool_status),
            "env_packages": {
                env: pkgs[:20]  # 只展示前20个
                for env, pkgs in self.env_packages.items()
            },
        }

    def print_status_report(self):
        """打印完整状态报告到控制台"""
        report = self.get_status_report()
        print("\n" + "=" * 70)
        print("  Conda Environment Status Report")
        print("=" * 70)
        print(f"  Conda binary: {report['conda_binary']}")
        print(f"  Available environments: {len(report['available_environments'])}")
        for env in report['available_environments']:
            print(f"    - {env}")
        print()
        print(f"  Tool Assignments:")
        for tool, env in report['tool_assignments'].items():
            status = report['tool_status'].get(tool, "?")
            print(f"    {tool:20s} -> {env:15s} [{status}]")
        print("=" * 70)


# ============================================================
# 模块级便捷函数
# ============================================================

# 全局单例（惰性初始化）
_global_conda_manager: Optional[CondaManager] = None


def get_conda_manager(config: Optional[Dict] = None) -> CondaManager:
    """
    获取全局 Conda 管理器单例

    Args:
        config: 配置字典（仅在首次调用时生效）

    Returns:
        CondaManager 实例
    """
    global _global_conda_manager
    if _global_conda_manager is None:
        _global_conda_manager = CondaManager(config=config)
        _global_conda_manager.scan_environments()
        _global_conda_manager.assign_tools()
    return _global_conda_manager


def reset_conda_manager():
    """重置全局 Conda 管理器（用于测试）"""
    global _global_conda_manager
    _global_conda_manager = None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # 快速自检
    cm = CondaManager()
    cm.scan_environments()
    cm.assign_tools()
    cm.print_status_report()

    # 测试在 RDKit 环境中执行
    if cm.is_tool_available("rdkit"):
        print("\n[Test] Running RDKit in assigned environment...")
        result = cm.run_python("rdkit", """
from rdkit import Chem
from rdkit.Chem import Descriptors
mol = Chem.MolFromSmiles("CCO")
print(f"Ethanol MW: {Descriptors.MolWt(mol):.2f}")
        """)
        print(result.stdout)

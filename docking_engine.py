#!/usr/bin/env python3
"""
模块: AutoDock-GPU 对接引擎
Module: AutoDock-GPU Docking Engine

借鉴服务器代码: ThreadPool 并行 + RANKING 表格能量解析 + 断点续传

工作流:
1. prepare_grid(ref_ligand) → GPF → AutoGrid4 → .maps.fld
2. batch_dock(df) → ThreadPool 并行 GPU 对接 → 解析结合能

用法:
    engine = DockingEngine(config={...})
    engine.prepare_grid(ref_ligand_pdbqt=Path("ligand.pdbqt"))
    results = engine.batch_dock(df, output_dir=Path("docking_output"))
"""

import logging
import os
import random
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import ADGPU_CONFIG, DIRS, TOOLS

logger = logging.getLogger(__name__)

# 并行 GPU 对接线程数（AutoDock-GPU 支持多任务交叠）
from config import ADGPU_CONFIG as _ADGPU_CFG
DOCK_WORKERS = _ADGPU_CFG.get("dock_workers", 8)


def parse_best_energy_dlg(dlg_path):
    """
    解析 AutoDock-GPU DLG 文件中的最佳结合能

    方法1: RANKING 表格行 — parts[0]=='1', parts[1]=='1' → parts[3]
    方法2: "Estimated Free Energy of Binding" 文本行正则
    """
    dlg_path = Path(dlg_path)
    if not dlg_path.exists():
        return None
    try:
        with open(dlg_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "RANKING" in line:
                    parts = " ".join(line.strip().split()).split(" ")
                    if len(parts) >= 5 and parts[0] == "1" and parts[1] == "1":
                        try:
                            return float(parts[3])
                        except ValueError:
                            continue
        # 回退：文本匹配
        with open(dlg_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            match = re.search(
                r"Estimated Free Energy of Binding\s*=\s*([-\d.]+)", content
            )
            if match:
                return float(match.group(1))
    except Exception as e:
        logger.debug(f"DLG parse error: {e}")
    return None


def parse_best_energy_xml(xml_path):
    """解析 AutoDock-GPU XML 中的最佳结合能"""
    xml_path = Path(xml_path)
    if not xml_path.exists():
        return None
    try:
        text = xml_path.read_text()
        energies = re.findall(r"<free_NRG_binding>\s*([-\d.]+)</free_NRG_binding>", text)
        if energies:
            return min(float(e) for e in energies)
    except Exception:
        pass
    return None


def parse_best_energy(output_dir, ligand_name):
    """自动检测 DLG 或 XML 并解析最佳结合能"""
    dlg = Path(output_dir) / f"{ligand_name}.dlg"
    xml = Path(output_dir) / f"{ligand_name}.xml"

    energy = parse_best_energy_dlg(dlg)
    if energy is None:
        energy = parse_best_energy_xml(xml)
    return energy


class DockingEngine:
    """
    AutoDock-GPU 对接引擎

    特性:
    - ThreadPool 并行 GPU 对接（默认 4 线程）
    - DLG RANKING 表格 + XML 双重能量解析
    - 断点续传：跳过已有成功结果的配体
    """

    def __init__(self, config: Optional[Dict] = None, conda_manager=None):
        self.config = {**ADGPU_CONFIG, **(config or {})}

        # 工具链路径
        self.adgpu_bin = TOOLS.get("autodock_gpu", "autodock_gpu_128wi")
        self.autogrid_bin = TOOLS.get("autogrid4", "autogrid4")
        self.prepare_gpf4_bin = TOOLS.get("prepare_gpf4", "prepare_gpf4")

        self._mock_mode = False
        self._grid_ready = False
        self._grid_fld: Optional[Path] = None

        if conda_manager is None:
            from conda_manager import get_conda_manager
            conda_manager = get_conda_manager()
        self.cm = conda_manager

        receptor_str = self.config.get("receptor", "")
        if receptor_str and Path(receptor_str).exists():
            self.receptor = Path(receptor_str).resolve()
        else:
            self.receptor = None
            if receptor_str:
                logger.warning(f"Receptor not found: {receptor_str}")

        self._check_tools()

    def _check_tools(self):
        missing = []
        # AutoDock-GPU: 先找二进制，不行就找 conda 环境
        self._use_conda_adgpu = False
        if not Path(self.adgpu_bin).exists():
            adgpu_env = self.cm.get_env("autodock_gpu")
            if adgpu_env and adgpu_env != "base":
                self._use_conda_adgpu = True
                self._adgpu_env = adgpu_env
            else:
                missing.append("AutoDock-GPU")
        # autogrid4 / prepare_gpf4 也可能在 adgpu conda 环境里
        if not Path(self.autogrid_bin).exists():
            ag_env = self.cm.get_env("autodock_gpu")
            ag_bin = Path.home() / "miniconda3" / "envs" / ag_env / "bin" / "autogrid4"
            if ag_bin.exists():
                self.autogrid_bin = str(ag_bin)
            else:
                missing.append("AutoGrid4")
        if not Path(self.prepare_gpf4_bin).exists():
            ag_env = self.cm.get_env("autodock_gpu")
            gpf_bin = Path.home() / "miniconda3" / "envs" / ag_env / "bin" / "prepare_gpf4"
            if gpf_bin.exists():
                self.prepare_gpf4_bin = str(gpf_bin)
            else:
                missing.append("prepare_gpf4")
        if missing:
            self._mock_mode = True
            logger.warning(f"Missing: {', '.join(missing)}. Using MOCK mode.")
        else:
            mode = f"conda:{self._adgpu_env}" if self._use_conda_adgpu else "direct"
            logger.info(f"Docking engine ready: AutoDock-GPU ({mode}) + AutoGrid4 + ADTools")

    # ============================================================
    # 网格地图
    # ============================================================

    def prepare_grid(self, ref_ligand_pdbqt: Path, output_dir=None, force=False) -> Path:
        """生成 AutoDock 网格地图（ADTools GPF + AutoGrid4）"""
        if self.receptor is None:
            raise RuntimeError("Receptor not set.")

        pdb_id = self.receptor.stem
        receptor_dir = self.receptor.parent
        output_dir = Path(output_dir or receptor_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fld_path = receptor_dir / f"{pdb_id}.maps.fld"

        # 检查 GPF 中的受体路径是否有效（防止从其他机器拷贝的网格）
        gpf_path = receptor_dir / f"{pdb_id}.gpf"
        if fld_path.exists() and not force and gpf_path.exists():
            gpf_text = gpf_path.read_text()
            if str(self.receptor.resolve()) in gpf_text or str(self.receptor.name) in gpf_text:
                logger.info(f"Grid maps exist: {fld_path}")
                self._grid_fld = fld_path
                self._grid_ready = True
                return fld_path
            else:
                logger.info(f"Grid maps have stale paths, regenerating...")

        cx, cy, cz = self.config["center_x"], self.config["center_y"], self.config["center_z"]
        nx, ny, nz = self.config.get("size_x", 40), self.config.get("size_y", 40), self.config.get("size_z", 40)
        spacing = self.config.get("grid_spacing", 0.375)

        import shutil
        local_ligand = receptor_dir / ref_ligand_pdbqt.name
        shutil.copy2(ref_ligand_pdbqt, local_ligand)

        cmd = [
            self.prepare_gpf4_bin,
            "-l", str(local_ligand.name), "-r", str(self.receptor.name),
            "-p", f"npts={nx},{ny},{nz}",
            "-p", f"gridcenter={cx:.3f},{cy:.3f},{cz:.3f}",
            "-p", f"spacing={spacing}",
            "-p", "ligand_types=A,C,HD,N,NA,OA,SA,F,CL,BR,I",
            "-o", f"{pdb_id}.gpf",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(receptor_dir))
        if r.returncode != 0:
            raise RuntimeError(f"prepare_gpf4 failed: {r.stderr[:500]}")

        r = subprocess.run(
            [self.autogrid_bin, "-p", f"{pdb_id}.gpf", "-l", f"{pdb_id}.glg"],
            capture_output=True, text=True, timeout=300, cwd=str(receptor_dir),
        )
        if r.returncode != 0 or not fld_path.exists():
            raise RuntimeError(f"AutoGrid4 failed: {r.stderr[:500]}")

        logger.info(f"Grid maps ready: {fld_path}")
        self._grid_fld = fld_path
        self._grid_ready = True
        return fld_path

    # ============================================================
    # 单分子对接
    # ============================================================

    def _dock_one(self, ligand_pdbqt: Path, output_dir: Path, ligand_name: str, timeout=300) -> dict:
        """运行单个 AutoDock-GPU 对接（自动选择 conda run 或直接调用）"""
        adgpu_args = [
            "--lfile", str(ligand_pdbqt.resolve()),
            "--ffile", str(self._grid_fld.resolve()),
            "--nrun", str(self.config.get("num_runs", 20)),
            "--resnam", str(output_dir.resolve() / ligand_name),
            "--xmloutput", "1",
            "--dlgoutput", "1",
            "--gbest", "1",
        ]
        if self._use_conda_adgpu:
            cmd = ["conda", "run", "-n", self._adgpu_env, self.adgpu_bin] + adgpu_args
        else:
            cmd = [self.adgpu_bin] + adgpu_args
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(output_dir))
            if r.returncode != 0:
                return {"success": False, "binding_energy": None, "error": f"rc={r.returncode}"}

            # AutoDock-GPU 可能把输出写到 fld 所在目录而非 output_dir
            energy = parse_best_energy(output_dir, ligand_name)
            if energy is None and self._grid_fld:
                energy = parse_best_energy(self._grid_fld.parent, ligand_name)
            return {"success": True, "binding_energy": energy, "error": None}
        except subprocess.TimeoutExpired:
            return {"success": False, "binding_energy": None, "error": "timeout"}
        except Exception as e:
            return {"success": False, "binding_energy": None, "error": str(e)}

    # ============================================================
    # 批量对接
    # ============================================================

    def batch_dock(self, df: pd.DataFrame, output_dir: Path, pdbqt_column="pdbqt_path") -> pd.DataFrame:
        """
        批量对接 — ThreadPool 并行 + 断点续传
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._mock_mode:
            return self._mock_batch_dock(df)

        if not self._grid_ready:
            ref = self._find_first_valid_ligand(df, pdbqt_column)
            if ref is None:
                return self._mock_batch_dock(df)
            self.prepare_grid(ref)

        # 收集任务 + 断点检测
        tasks = []
        pre_ok_map = {}
        for idx, row in df.iterrows():
            p = row.get(pdbqt_column, "")
            if not p or not Path(p).exists():
                continue
            name = Path(p).stem
            dlg = output_dir / f"{name}.dlg"
            xml = output_dir / f"{name}.xml"

            # 断点续传：已有有效结果则跳过
            existing = parse_best_energy(output_dir, name)
            if existing is not None:
                pre_ok_map[name] = existing
                continue
            tasks.append((idx, Path(p), name))

        n_skip = len(pre_ok_map)
        if n_skip > 0:
            logger.info(f"  Resuming: {n_skip} already docked, {len(tasks)} remaining")

        # ThreadPool 并行对接
        ok_map = dict(pre_ok_map)
        if tasks:
            logger.info(f"  Docking {len(tasks)} ligands (workers={DOCK_WORKERS})...")
            completed = 0
            with ThreadPoolExecutor(max_workers=DOCK_WORKERS) as executor:
                futures = {
                    executor.submit(self._dock_one, t[1], output_dir, t[2]): t
                    for t in tasks
                }
                for f in as_completed(futures):
                    idx, pdbqt, name = futures[f]
                    result = f.result()
                    if result["success"] and result["binding_energy"] is not None:
                        ok_map[name] = result["binding_energy"]
                    elif result["success"]:
                        ok_map[name] = None  # docked but no energy parsed
                    completed += 1
                    if completed % 20 == 0:
                        logger.info(f"    Docking: {completed}/{len(tasks)}")

        # 构建结果
        results = []
        for idx, row in df.iterrows():
            p = row.get(pdbqt_column, "")
            name = Path(p).stem if p else f"ligand_{idx}"
            energy = ok_map.get(name)

            row_data = row.to_dict()
            row_data["docking_success"] = energy is not None
            row_data["binding_energy"] = energy
            row_data["docking_error"] = None if energy is not None else "Failed"
            results.append(row_data)

        result_df = pd.DataFrame(results)
        n_ok = result_df["docking_success"].sum()
        logger.info(f"  Docking done: {n_ok}/{len(result_df)} success")
        return result_df

    def _find_first_valid_ligand(self, df, col):
        for _, row in df.iterrows():
            p = row.get(col, "")
            if p and Path(p).exists():
                return Path(p)
        return None

    def _mock_batch_dock(self, df):
        results = []
        for _, row in df.iterrows():
            d = row.to_dict()
            d.update({"docking_success": True, "binding_energy": round(-random.uniform(6, 12), 2),
                       "docking_error": None, "_mock": True})
            results.append(d)
        return pd.DataFrame(results)

    # ============================================================
    # 配置
    # ============================================================

    def set_receptor(self, path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        self.receptor = p.resolve()
        self.config["receptor"] = str(p)
        self._grid_ready = False

    def set_grid_from_dict(self, grid_data):
        c, s = grid_data.get("center"), grid_data.get("size")
        if c is not None:
            self.config["center_x"], self.config["center_y"], self.config["center_z"] = float(c[0]), float(c[1]), float(c[2])
        if s is not None:
            sp = self.config.get("grid_spacing", 0.375)
            self.config["size_x"] = max(20, int(s[0] / sp))
            self.config["size_y"] = max(20, int(s[1] / sp))
            self.config["size_z"] = max(20, int(s[2] / sp))

    @property
    def mock_mode(self):
        return self._mock_mode


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    e = DockingEngine()
    print(f"AD-GPU: {Path(e.adgpu_bin).exists()}")
    print(f"AutoGrid4: {Path(e.autogrid_bin).exists()}")
    print(f"Mock: {e._mock_mode}")

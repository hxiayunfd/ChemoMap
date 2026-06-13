#!/usr/bin/env python3
"""
模块: 蛋白准备器
Module: Protein Preparator — PDB Download, Fix, PDBQT Conversion

完整的蛋白准备流水线:
1. 从 RCSB PDB 下载原始 PDB 文件
2. 扫描共结晶配体 → 检测结合口袋
3. 交互式口袋选择（用户确认网格中心/大小）
4. PDBFixer 修复蛋白结构（补缺失残基/原子、加氢）
5. 非极性氢裁剪（仅保留极性氢）
6. prepare_receptor4 转换为 PDBQT 格式

工具依赖:
    - PDBFixer / OpenMM: conda env "pdbfixer"
    - requests: conda env "requests"
    - prepare_receptor4: 独立可执行文件（AutoDock Tools）

用法:
    from protein_preparator import ProteinPreparator

    prep = ProteinPreparator()
    # 交互式全流程
    result = prep.prepare_protein_interactive("4dkl")
    # 或分步执行
    pdb_path = prep.download_pdb("4dkl")
    pockets = prep.detect_binding_pockets(pdb_path)
    pdbqt = prep.fix_and_prepare_protein(pdb_path, "4dkl")
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from config import DIRS, CONDA_TOOLS, TOOLS

logger = logging.getLogger(__name__)

# 结晶辅剂/溶剂/离子/糖基化黑名单
LIGAND_BLACKLIST: set = {
    "HOH", "DOD", "WAT", "SOL",
    "SO4", "PO4", "NO3", "CO3", "EDT", "ACT", "FOR", "AET",
    "CL", "NA", "MG", "ZN", "CA", "K", "NI", "CU", "MN", "FE",
    "GOL", "EDO", "PEG", "PG4", "PGE", "PEO", "DTT", "DMS", "BME",
    "TMS", "NAG", "MAN", "BMA", "FUC", "GAL",
}


class ProteinPreparator:
    """
    蛋白准备器

    一站式蛋白准备：下载 → 口袋检测 → 修复 → PDBQT 转换。

    网络请求操作通过 conda run 在对应环境中执行。
    PDBFixer/OpenMM 通过 conda run 调用。
    prepare_receptor4 作为独立可执行文件直接调用。
    """

    def __init__(self, conda_manager=None):
        """
        Args:
            conda_manager: CondaManager 实例（可选，自动获取全局实例）
        """
        if conda_manager is None:
            from conda_manager import get_conda_manager
            conda_manager = get_conda_manager()
        self.cm = conda_manager

    # ============================================================
    # 子进程辅助
    # ============================================================

    def _run_python(self, tool: str, script: str, timeout: int = 600) -> str:
        """在指定工具的 conda 环境中运行 Python 脚本"""
        result = self.cm.run_python(tool, script, timeout=timeout)
        return result.stdout

    # ============================================================
    # PDB 下载
    # ============================================================

    def download_pdb(
        self,
        pdb_id: str,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        从 RCSB PDB 下载原始 PDB 文件

        Args:
            pdb_id: 4 字符 PDB 编号（如 "4dkl"）
            output_dir: 输出目录

        Returns:
            下载的 PDB 文件路径，失败返回 None
        """
        pdb_id = pdb_id.lower()
        output_dir = Path(output_dir or DIRS["raw_pdb"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{pdb_id}.pdb"

        if output_path.exists():
            logger.info(f"PDB file already exists: {output_path}")
            return output_path

        logger.info(f"Downloading PDB {pdb_id.upper()} from RCSB...")

        script = f"""
import json, requests
url = "https://files.rcsb.org/download/{pdb_id}.pdb"
try:
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        with open({str(output_path)!r}, "w", encoding="utf-8") as f:
            f.write(r.text)
        print(json.dumps({{"success": True, "path": {str(output_path)!r}}}))
    else:
        print(json.dumps({{"success": False, "error": f"HTTP {{r.status_code}}"}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
"""
        try:
            output = self._run_python("requests", script, timeout=60)
            result = json.loads(output.strip())
            if result.get("success"):
                logger.info(f"Downloaded: {output_path}")
                return output_path
            else:
                logger.error(f"Download failed: {result.get('error')}")
                return None
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None

    # ============================================================
    # 结合口袋检测
    # ============================================================

    def detect_binding_pockets(
        self,
        pdb_path: Path,
        padding: float = 12.0,
    ) -> List[Dict[str, Any]]:
        """
        扫描 PDB 中所有共结晶配体并计算口袋网格参数

        读取 PDB 的 HETATM 行，过滤溶剂/离子/辅因子，
        对每个共结晶配体计算其几何中心和包围盒。

        Args:
            pdb_path: PDB 文件路径
            padding: 网格扩展尺寸 (Å)，默认 12.0

        Returns:
            口袋候选列表:
            [{"name": "A_ABC_301", "center": np.array([x,y,z]),
              "size": np.array([sx,sy,sz]), "info": "..."}, ...]
        """
        all_ligands: Dict[str, List[List[float]]] = {}

        with open(pdb_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.startswith("HETATM"):
                    continue

                res_name = line[17:20].strip()
                if res_name in LIGAND_BLACKLIST:
                    continue

                chain_id = line[21:22].strip() or "A"
                res_seq = line[22:26].strip()
                ligand_key = f"{chain_id}_{res_name}_{res_seq}"

                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError:
                    continue

                all_ligands.setdefault(ligand_key, []).append([x, y, z])

        candidates = []

        if not all_ligands:
            logger.info("No co-crystallized ligands found. Using blind docking.")
            # 全蛋白几何中心
            protein_coords = []
            with open(pdb_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith("ATOM"):
                        try:
                            protein_coords.append([
                                float(line[30:38]),
                                float(line[38:46]),
                                float(line[46:54]),
                            ])
                        except ValueError:
                            continue

            if protein_coords:
                coords = np.array(protein_coords)
                center = np.mean(coords, axis=0)
                candidates.append({
                    "name": "Global_Blind_Docking",
                    "center": center,
                    "size": np.array([30.0, 30.0, 30.0]),
                    "info": "全蛋白几何中心（盲对接）",
                })
            return candidates

        # 每个配体生成一个口袋
        for lig_key, points in all_ligands.items():
            coords = np.array(points)
            center = np.mean(coords, axis=0)
            min_bounds = np.min(coords, axis=0)
            max_bounds = np.max(coords, axis=0)
            ligand_sizes = max_bounds - min_bounds

            candidates.append({
                "name": lig_key,
                "center": center,
                "size": ligand_sizes + padding,
                "info": f"共结晶配体 ({len(points)} 重原子)",
            })

        logger.info(f"Detected {len(candidates)} binding pocket(s)")
        return candidates

    # ============================================================
    # Grid 文件导出
    # ============================================================

    def export_grid_file(
        self,
        grid_data: Dict[str, Any],
        pdb_id: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        将口袋参数导出为 Grid 配置文件（文本格式）

        输出格式兼容 AutoDock-GPU 参数。

        Args:
            grid_data: 口袋数据 (name, center, size)
            pdb_id: PDB 编号
            output_dir: 输出目录

        Returns:
            生成的 Grid 文件路径
        """
        pdb_id = pdb_id.lower()
        output_dir = Path(output_dir or DIRS["grid"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{pdb_id}.txt"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Auto-generated Grid Box for {pdb_id.upper()}\n")
            f.write(f"# Source: {grid_data['name']}\n")
            f.write(f"center_x = {grid_data['center'][0]:.3f}\n")
            f.write(f"center_y = {grid_data['center'][1]:.3f}\n")
            f.write(f"center_z = {grid_data['center'][2]:.3f}\n")
            f.write(f"size_x = {grid_data['size'][0]:.3f}\n")
            f.write(f"size_y = {grid_data['size'][1]:.3f}\n")
            f.write(f"size_z = {grid_data['size'][2]:.3f}\n")

        logger.info(f"Grid file exported: {output_path}")
        return output_path

    def read_grid_file(self, grid_path: Path) -> Dict[str, Any]:
        """
        从 Grid 配置文件读取口袋参数

        Args:
            grid_path: Grid 文本文件路径

        Returns:
            {"center_x": float, "center_y": ..., "size_x": ...}
        """
        params = {}
        with open(grid_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    params[key.strip()] = float(val.strip())
        return params

    # ============================================================
    # 蛋白修复 + PDBQT 转换
    # ============================================================

    def fix_and_prepare_protein(
        self,
        input_path: Path,
        pdb_id: str,
        output_dir: Optional[Path] = None,
        remove_nonpolar_h: bool = True,
        ph: float = 7.0,
    ) -> Optional[Path]:
        """
        修复蛋白结构并转换为 PDBQT 格式

        步骤:
        1. PDBFixer 修复（去水、补残基/原子、加氢）
        2. 非极性氢裁剪
        3. prepare_receptor4 转换为 PDBQT

        Args:
            input_path: 输入 PDB 文件路径
            pdb_id: PDB 编号
            output_dir: 输出目录
            remove_nonpolar_h: 是否裁剪非极性氢
            ph: 加氢 pH 值

        Returns:
            PDBQT 文件路径，失败返回 None
        """
        pdb_id = pdb_id.lower()
        output_dir = Path(output_dir or DIRS["prepared_protein"])
        output_dir.mkdir(parents=True, exist_ok=True)
        final_pdbqt = output_dir / f"{pdb_id}.pdbqt"

        if final_pdbqt.exists():
            logger.info(f"PDBQT already exists: {final_pdbqt}")
            return final_pdbqt

        temp_pdb = output_dir / f"{pdb_id}_fixed.pdb"

        logger.info(f"Fixing protein {pdb_id.upper()} with PDBFixer...")

        # Step 1-2: PDBFixer + 非极性氢裁剪
        fix_script = f"""
import json, sys, os

try:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    PDBFIXER_OK = True
except ImportError:
    PDBFIXER_OK = False

input_path = {str(input_path)!r}
temp_pdb = {str(temp_pdb)!r}
ph = {ph}
remove_h = {str(remove_nonpolar_h).lower() == 'true'}

if not PDBFIXER_OK:
    print(json.dumps({{"success": False, "error": "PDBFixer/OpenMM not installed"}}))
    sys.exit(0)

try:
    fixer = PDBFixer(filename=input_path)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    with open(temp_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    if remove_h:
        # 极性氢白名单过滤
        POLAR_H = {{
            "H", "H1", "H2", "H3",
            "HE", "HH11", "HH12", "HH21", "HH22",
            "HD21", "HD22", "HE21", "HE22",
            "HD1", "HE2", "HE1",
            "HZ1", "HZ2", "HZ3",
            "HG", "HH",
        }}
        POLAR_H_RES = {{
            "ARG": ["HE","HH11","HH12","HH21","HH22"],
            "ASN": ["HD21","HD22"],
            "GLN": ["HE21","HE22"],
            "HIS": ["HD1","HE2"],
            "TRP": ["HE1"],
            "LYS": ["HZ1","HZ2","HZ3"],
            "SER": ["HG"], "THR": ["HG"], "CYS": ["HG"],
            "TYR": ["HH"],
        }}

        clean = []
        with open(temp_pdb) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    aname = line[12:16].strip()
                    rname = line[17:20].strip()
                    if "H" in aname:
                        if aname in POLAR_H:
                            clean.append(line); continue
                        if rname in POLAR_H_RES and any(x in aname for x in POLAR_H_RES[rname]):
                            clean.append(line); continue
                        continue
                clean.append(line)
        with open(temp_pdb, 'w') as f:
            f.writelines(clean)

    print(json.dumps({{"success": True, "temp_pdb": temp_pdb}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
"""
        try:
            output = self._run_python("pdbfixer", fix_script, timeout=600)
            result = json.loads(output.strip())
            if not result.get("success"):
                logger.error(f"PDBFixer failed: {result.get('error')}")
                return None
        except Exception as e:
            logger.error(f"PDBFixer script error: {e}")
            return None

        # Step 3: PDB → PDBQT
        success = self._convert_to_pdbqt(temp_pdb, final_pdbqt)

        # 清理临时文件
        if temp_pdb.exists():
            temp_pdb.unlink()

        if success:
            logger.info(f"Protein PDBQT ready: {final_pdbqt}")
            return final_pdbqt
        return None

    def _convert_to_pdbqt(
        self,
        pdb_path: Path,
        pdbqt_path: Path,
    ) -> bool:
        """
        使用 prepare_receptor4 将 PDB 转换为 PDBQT

        Args:
            pdb_path: 输入 PDB 文件
            pdbqt_path: 输出 PDBQT 文件

        Returns:
            是否成功
        """
        # 查找 prepare_receptor4
        prep4 = TOOLS.get("prepare_receptor4", "prepare_receptor4")
        candidates = [prep4]
        if not os.path.exists(str(prep4)):
            candidates = [
                str(Path.home() / "miniconda3" / "bin" / "prepare_receptor4"),
                "prepare_receptor4",
                "prepare_receptor4.py",
            ]

        cmd_path = None
        for c in candidates:
            if os.path.exists(c) or __import__('shutil').which(c):
                cmd_path = c
                break

        if not cmd_path:
            logger.error(
                "prepare_receptor4 not found. "
                "Install AutoDock Tools or set TOOLS['prepare_receptor4'] in config."
            )
            return False

        cmd = [
            cmd_path,
            "-r", str(pdb_path),
            "-o", str(pdbqt_path),
            "-A", "checkhydrogens",
            "-U", "nphs_lps_waters",
        ]

        logger.info(f"Converting to PDBQT: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                logger.error(f"prepare_receptor4 failed: {result.stderr[:500]}")
                return False
            return pdbqt_path.exists()
        except subprocess.TimeoutExpired:
            logger.error("PDBQT conversion timed out")
            return False
        except Exception as e:
            logger.error(f"PDBQT conversion error: {e}")
            return False

    # ============================================================
    # 交互式口袋选择
    # ============================================================

    def interactive_pocket_selection(
        self,
        pdb_path: Path,
        padding: float = 12.0,
    ) -> Optional[Dict[str, Any]]:
        """
        交互式口袋选择

        扫描所有共结晶配体 → 展示给用户 → 用户确认/修改坐标和网格大小

        Args:
            pdb_path: PDB 文件路径
            padding: 网格扩展尺寸 (Å)

        Returns:
            用户确认的口袋参数，或 None（用户取消）
        """
        pockets = self.detect_binding_pockets(pdb_path, padding)

        print("\n" + "=" * 70)
        print("  结合口袋扫描结果 / Binding Pocket Detection Results")
        print("=" * 70)

        if not pockets:
            print("  ⚠ 未检测到任何配体")
            return None

        print(f"\n  检测到 {len(pockets)} 个候选口袋:\n")
        for i, pocket in enumerate(pockets):
            c = pocket["center"]
            s = pocket["size"]
            print(f"  [{i + 1}] {pocket['name']}")
            print(f"      {pocket['info']}")
            print(f"      中心: X={c[0]:.3f}  Y={c[1]:.3f}  Z={c[2]:.3f}")
            print(f"      大小: X={s[0]:.3f}  Y={s[1]:.3f}  Z={s[2]:.3f}")
            print()

        # 用户选择
        while True:
            try:
                choice = input(
                    f"\n  请选择目标口袋 (1-{len(pockets)}, 回车=第1个): "
                ).strip()
                if choice == "":
                    selected_idx = 0
                else:
                    selected_idx = int(choice) - 1
                    if not (0 <= selected_idx < len(pockets)):
                        print(f"  ⚠ 请输入 1-{len(pockets)}")
                        continue
                break
            except ValueError:
                print(f"  ⚠ 请输入数字 (1-{len(pockets)})")
                continue

        selected = pockets[selected_idx]
        print(f"\n  ✅ 已选择: {selected['name']}")

        # 确认/修改中心坐标
        center = selected["center"].copy()
        print("\n  --- 中心坐标确认 ---")
        print(f"  当前: X={center[0]:.3f}  Y={center[1]:.3f}  Z={center[2]:.3f}")
        for idx, axis in enumerate(["X", "Y", "Z"]):
            val = input(f"  中心 {axis} (回车={center[idx]:.3f}): ").strip()
            if val:
                try:
                    center[idx] = float(val)
                except ValueError:
                    print(f"  ⚠ 无效输入，保持 {center[idx]:.3f}")

        # 确认/修改网格大小
        size = selected["size"].copy()
        print("\n  --- 网格大小确认 ---")
        print(f"  当前: X={size[0]:.3f}  Y={size[1]:.3f}  Z={size[2]:.3f}")
        for idx, axis in enumerate(["X", "Y", "Z"]):
            val = input(f"  大小 {axis} (回车={size[idx]:.3f}): ").strip()
            if val:
                try:
                    size[idx] = float(val)
                except ValueError:
                    print(f"  ⚠ 无效输入，保持 {size[idx]:.3f}")

        # 最终确认
        print("\n  --- 最终参数 ---")
        print(f"  口袋: {selected['name']}")
        print(f"  中心: X={center[0]:.3f}  Y={center[1]:.3f}  Z={center[2]:.3f}")
        print(f"  大小: X={size[0]:.3f}  Y={size[1]:.3f}  Z={size[2]:.3f}")

        if input("\n  确认? (Y/n): ").strip().lower() in ("n", "no"):
            print("  ❌ 已取消")
            return None

        print("  ✅ 已确认\n" + "=" * 70)
        return {
            "name": selected["name"],
            "center": center,
            "size": size,
            "info": selected["info"],
        }

    # ============================================================
    # 一站式蛋白准备流水线
    # ============================================================

    def prepare_protein_pipeline(
        self,
        pdb_id: str,
        output_dir: Optional[Path] = None,
        padding: float = 12.0,
        remove_nonpolar_h: bool = True,
        ph: float = 7.0,
        interactive: bool = False,
    ) -> Dict[str, Any]:
        """
        一站式蛋白准备流水线

        步骤:
        1. 下载 PDB
        2. 检测结合口袋
        3. (可选) 交互式选择口袋
        4. 导出 Grid 文件
        5. 修复 + PDBQT 转换

        Args:
            pdb_id: PDB 编号
            output_dir: 输出根目录
            padding: 网格扩展
            remove_nonpolar_h: 是否裁剪非极性氢
            ph: 加氢 pH
            interactive: 是否交互式选择口袋

        Returns:
            {
                "pdb_id": str,
                "pdb_path": Path or None,
                "pdbqt_path": Path or None,
                "grid_path": Path or None,
                "pockets": [...],
                "selected_pocket": {...} or None,
            }
        """
        pdb_id = pdb_id.lower()
        output_dir = Path(output_dir or DIRS["output"])

        result = {
            "pdb_id": pdb_id,
            "pdb_path": None,
            "pdbqt_path": None,
            "grid_path": None,
            "pockets": [],
            "selected_pocket": None,
        }

        # 1. 下载
        pdb_path = self.download_pdb(pdb_id, output_dir / "raw_pdb")
        if not pdb_path:
            logger.error(f"Failed to download PDB {pdb_id}")
            return result
        result["pdb_path"] = pdb_path

        # 2. 检测口袋
        pockets = self.detect_binding_pockets(pdb_path, padding)
        result["pockets"] = pockets

        # 3. 交互式选择
        selected = None
        if interactive and pockets:
            selected = self.interactive_pocket_selection(pdb_path, padding)
        elif pockets:
            selected = pockets[0]  # 自动选第一个

        result["selected_pocket"] = selected

        # 4. 导出 Grid
        if selected:
            grid_path = self.export_grid_file(selected, pdb_id, output_dir / "grid")
            result["grid_path"] = grid_path

        # 5. 修复 + PDBQT
        pdbqt_path = self.fix_and_prepare_protein(
            pdb_path, pdb_id,
            output_dir / "prepared_protein",
            remove_nonpolar_h=remove_nonpolar_h,
            ph=ph,
        )
        result["pdbqt_path"] = pdbqt_path

        return result

    def prepare_protein_interactive(self, pdb_id: str) -> Dict[str, Any]:
        """
        交互式蛋白准备（快捷方法）

        等同于 prepare_protein_pipeline(interactive=True)
        """
        return self.prepare_protein_pipeline(pdb_id, interactive=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    prep = ProteinPreparator()

    # 测试口袋检测
    test_dir = DIRS["raw_pdb"]
    if test_dir.exists():
        for pdb_file in test_dir.glob("*.pdb"):
            pockets = prep.detect_binding_pockets(pdb_file)
            print(f"\n{pdb_file.name}: {len(pockets)} pocket(s)")
            for p in pockets:
                print(f"  - {p['name']}: center={p['center']}, size={p['size']}")

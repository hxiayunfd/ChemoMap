#!/usr/bin/env python3
"""
模块: 配体准备器
Module: Ligand Preparator — SMILES → PDBQT

直接调用 OpenBabel: obabel -ismi -opdbqt --gen3D --partialcharge gasteiger

用法:
    from ligand_preparator import LigandPreparator
    prep = LigandPreparator()
    prep.batch_convert_to_pdbqt(df, output_dir)
"""

import logging
import os
import subprocess
import sys
from multiprocessing import Pool, Value, Lock
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DIRS

logger = logging.getLogger(__name__)

# 进程数
MAX_WORKERS = 20

# 全局共享计数器（多进程安全）
_counter = None
_total = 0
_print_lock = Lock()


def _init_worker(counter, total):
    global _counter, _total
    _counter = counter
    _total = total


def _convert_one(args):
    """单个分子: obabel SMILES → PDBQT"""
    global _counter, _total
    compound_name, smiles, output_dir = args
    output_path = os.path.join(output_dir, f"{compound_name}.pdbqt")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        with _counter.get_lock():
            _counter.value += 1
        return compound_name, True

    import shutil
    obabel = shutil.which("obabel") or "obabel"
    cmd = [
        obabel, "-ismi", "-opdbqt",
        "-O", output_path,
        "--gen3D", "--partialcharge", "gasteiger",
    ]

    try:
        p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        p.communicate(input=f"{smiles} {compound_name}", timeout=30)
        ok = os.path.exists(output_path) and os.path.getsize(output_path) > 100

        with _counter.get_lock():
            _counter.value += 1

        if _counter.value % 50 == 0 or _counter.value == _total:
            with _print_lock:
                sys.stdout.write(
                    f"\r  PDBQT: {_counter.value}/{_total} "
                    f"({_counter.value * 100 // _total}%)\n"
                )
                sys.stdout.flush()

        return compound_name, ok
    except Exception:
        with _counter.get_lock():
            _counter.value += 1
        return compound_name, False


class LigandPreparator:
    """配体准备器 — OpenBabel 直接 SMILES → PDBQT"""

    def __init__(self, conda_manager=None):
        from config import TOOLS as _cfg_tools
        self._obabel_bin = _cfg_tools.get("obabel", "obabel")
        self._has_obabel = True
        try:
            subprocess.run([self._obabel_bin, "-V"], capture_output=True, timeout=5)
        except Exception:
            self._has_obabel = False
            logger.warning(f"OpenBabel not found at '{self._obabel_bin}'!")

    def batch_convert_to_pdbqt(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        smiles_column: str = "SMILES",
        name_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        批量 SMILES → PDBQT (OpenBabel 多进程并行)

        obabel -ismi -opdbqt --gen3D --partialcharge gasteiger
        200 分子 ~5 秒 (20 并行)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        n_total = len(df)

        tasks = []
        for idx, row in df.iterrows():
            smiles = row[smiles_column]
            base_name = (
                f"{row[name_column]}_{idx}"
                if name_column and name_column in row
                else f"ligand_{idx}"
            )
            safe_name = "".join(
                c if c.isalnum() or c in "_-" else "_"
                for c in str(base_name)
            )
            tasks.append((safe_name, smiles, str(output_dir)))

        logger.info(
            f"  PDBQT: {n_total} SMILES → PDBQT "
            f"(obabel, {MAX_WORKERS} workers)..."
        )

        shared_counter = Value("i", 0)
        ok_map = {}

        with Pool(
            processes=MAX_WORKERS,
            initializer=_init_worker,
            initargs=(shared_counter, n_total),
        ) as pool:
            for name, ok in pool.imap_unordered(_convert_one, tasks):
                ok_map[name] = ok

        # 构建结果
        results = []
        for idx, row in df.iterrows():
            base_name = (
                f"{row[name_column]}_{idx}"
                if name_column and name_column in row
                else f"ligand_{idx}"
            )
            safe_name = "".join(
                c if c.isalnum() or c in "_-" else "_"
                for c in str(base_name)
            )
            pdbqt_path = str(output_dir / f"{safe_name}.pdbqt")
            ok = ok_map.get(safe_name, False)

            row_data = row.to_dict()
            row_data["pdbqt_path"] = pdbqt_path if ok else None
            row_data["pdbqt_exists"] = ok
            results.append(row_data)

        result_df = pd.DataFrame(results)
        n_ok = result_df["pdbqt_exists"].sum()
        logger.info(f"  PDBQT done: {n_ok}/{n_total} success")
        return result_df

    # 旧接口兼容
    def batch_convert_to_sdf(self, df, output_dir, smiles_column="SMILES", name_column=None):
        return self.batch_convert_to_pdbqt(df, output_dir, smiles_column, name_column)

    def batch_convert_sdf_to_pdbqt(self, df, output_dir, sdf_column="sdf_path"):
        if "pdbqt_exists" in df.columns:
            return df
        logger.error("Use batch_convert_to_pdbqt() instead")
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    prep = LigandPreparator()
    print(f"OpenBabel: {prep._has_obabel}")

    test_df = pd.DataFrame({
        "SMILES": ["c1ccccc1", "CCO", "CNC1(C2=CC=CC=C2Cl)CCCCC1=O"] * 10,
    })

    import time
    out = DIRS["temp"] / "ligand_test4"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = prep.batch_convert_to_pdbqt(test_df, out)
    elapsed = time.time() - t0
    n_ok = result["pdbqt_exists"].sum()
    print(f"\n{len(test_df)} mols → {n_ok} PDBQT in {elapsed:.1f}s ({len(test_df)/elapsed:.0f}/s)")

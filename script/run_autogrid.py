#!/usr/bin/env python3
"""
AutoGrid4 网格地图生成脚本

功能:
1. 使用 AutoDockTools (GridParameter4FileMaker) 生成 GPF 参数文件
2. 调用 autogrid4 生成 .map 网格地图文件

用法:
  python run_autogrid.py --receptor <蛋白.pdbqt> --ligand <配体.pdbqt> \\
                         --center_x <x> --center_y <y> --center_z <z> \\
                         --size_x <x> --size_y <y> --size_z <z> \\
                         [--output_dir <目录>] [--spacing <间距>]

示例:
  python run_autogrid.py \\
      --receptor ../output/prepared_protein/4dkl.pdbqt \\
      --ligand ../output/rounds/round_001/03_pdbqt/ligand_0.pdbqt \\
      --center_x -27.997 --center_y -13.204 --center_z -11.084 \\
      --size_x 17.642 --size_y 23.867 --size_z 20.299 \\
      --output_dir ../output/grid_test

也可以从口袋参数文件 (.txt) 读取网格参数:
  python run_autogrid.py --receptor ../output/prepared_protein/4dkl.pdbqt \\
      --ligand ../output/rounds/round_001/03_pdbqt/ligand_0.pdbqt \\
      --param_file ../output/grid/4dkl.txt \\
      --output_dir ../output/grid_test
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 默认原子类型 (AutoDock4 标准)
DEFAULT_LIGAND_TYPES = ["A", "C", "HD", "N", "NA", "OA", "SA", "F", "CL", "BR", "I"]
DEFAULT_SPACING = 0.375


def find_autogrid4() -> Optional[str]:
    """查找 autogrid4 可执行文件"""
    # 优先检查系统路径
    autogrid_path = shutil.which("autogrid4")
    if autogrid_path:
        return autogrid_path

    # 常见安装路径
    common_paths = [
        "/usr/bin/autogrid4",
        "/usr/local/bin/autogrid4",
        "/opt/autogrid4/bin/autogrid4",
    ]
    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def parse_param_file(param_file: Path) -> Dict[str, float]:
    """
    解析口袋参数文件 (.txt)

    文件格式示例:
        center_x = -27.997
        center_y = -13.204
        center_z = -11.084
        size_x = 17.642
        size_y = 23.867
        size_z = 20.299

    Returns:
        包含 center_x/y/z 和 size_x/y/z 的字典
    """
    params = {}
    with open(param_file, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key.startswith("center_") or key.startswith("size_"):
                    try:
                        params[key] = float(value)
                    except ValueError:
                        logger.warning(f"无法解析参数: {key}={value}")

    required_keys = ["center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]
    missing = [k for k in required_keys if k not in params]
    if missing:
        logger.error(f"参数文件缺少必要参数: {missing}")
        sys.exit(1)

    logger.info(f"从参数文件读取网格参数: {params}")
    return params


def generate_gpf_with_adt(
    receptor_pdbqt: Path,
    ligand_pdbqt: Optional[Path],
    grid_params: Dict[str, float],
    output_gpf: Path,
    spacing: float = DEFAULT_SPACING,
    ligand_types: Optional[List[str]] = None,
) -> Path:
    """
    使用 AutoDockTools (GridParameter4FileMaker) Python API 生成 GPF 文件

    Args:
        receptor_pdbqt: 受体 PDBQT 文件路径
        ligand_pdbqt: 配体 PDBQT 文件路径（可选，用于自动检测原子类型）
        grid_params: 网格参数字典 (center_x/y/z, size_x/y/z)
        output_gpf: 输出 GPF 文件路径
        spacing: 网格间距（默认 0.375 Å）
        ligand_types: 手动指定的原子类型列表（当没有配体文件时使用）

    Returns:
        GPF 文件路径
    """
    from AutoDockTools.GridParameters import GridParameter4FileMaker

    # 计算网格点数 (确保为偶数)
    npts_x = int((grid_params["size_x"] / spacing + 0.5) // 2 * 2)
    npts_y = int((grid_params["size_y"] / spacing + 0.5) // 2 * 2)
    npts_z = int((grid_params["size_z"] / spacing + 0.5) // 2 * 2)

    # 确保网格点数在合理范围内
    npts_x = max(min(npts_x, 120), 20)
    npts_y = max(min(npts_y, 120), 20)
    npts_z = max(min(npts_z, 120), 20)

    logger.info(f"网格点数: npts=({npts_x}, {npts_y}, {npts_z})")
    logger.info(f"网格中心: ({grid_params['center_x']:.3f}, {grid_params['center_y']:.3f}, {grid_params['center_z']:.3f})")
    logger.info(f"网格间距: {spacing:.3f} Å")

    # 重要: 先解析所有路径（在 chdir 之前），避免 chdir 后 resolve() 解析到错误目录
    receptor_path = Path(receptor_pdbqt).resolve()
    receptor_dir = receptor_path.parent
    output_gpf_abs = str(Path(output_gpf).resolve())
    original_cwd = Path.cwd()

    # 创建 GridParameter4FileMaker 实例
    gpf_maker = GridParameter4FileMaker(verbose=False)

    try:
        # 重要: AutoDockTools 的 set_receptor/set_receptor4 内部使用 MolKit.Read()
        # 读取文件时依赖当前工作目录。如果文件不在当前目录，即使传入绝对路径，
        # set_receptor4 也会因为只使用 basename 而找不到文件。
        # 因此需要临时切换到受体文件所在目录。
        os.chdir(str(receptor_dir))
        logger.info(f"临时切换工作目录到: {receptor_dir}")

        # 设置受体（使用相对路径，因为已切换到文件所在目录）
        gpf_maker.set_receptor(receptor_path.name)
        logger.info(f"受体文件: {receptor_path}")

        # 设置配体（可选，用于自动检测原子类型）
        if ligand_pdbqt and Path(ligand_pdbqt).exists():
            ligand_path = Path(ligand_pdbqt).resolve()
            ligand_dir = ligand_path.parent
            # 如果配体在同一目录，使用相对路径；否则使用绝对路径
            if ligand_dir == receptor_dir:
                gpf_maker.set_ligand(ligand_path.name)
            else:
                gpf_maker.set_ligand(str(ligand_path))
            logger.info(f"配体文件: {ligand_path}（用于原子类型检测）")
        else:
            # 手动指定原子类型
            types = ligand_types or DEFAULT_LIGAND_TYPES
            ligand_types_str = " ".join(types)
            gpf_maker.set_grid_parameters(ligand_types=ligand_types_str)
            logger.info(f"手动指定原子类型: {ligand_types_str}")

        # 设置网格参数
        npts_str = f"{npts_x},{npts_y},{npts_z}"
        gridcenter_str = (
            f"{grid_params['center_x']:.3f},"
            f"{grid_params['center_y']:.3f},"
            f"{grid_params['center_z']:.3f}"
        )
        gpf_maker.set_grid_parameters(
            npts=npts_str,
            gridcenter=gridcenter_str,
            spacing=spacing,
        )

        # 写入 GPF 文件（使用预先解析的绝对路径）
        gpf_maker.write_gpf(output_gpf_abs)
        logger.info(f"GPF 文件已生成: {output_gpf_abs}")

    finally:
        # 恢复原始工作目录
        os.chdir(str(original_cwd))
        logger.info(f"恢复工作目录到: {original_cwd}")

    return output_gpf


def run_autogrid4(
    gpf_file: Path,
    log_file: Path,
    work_dir: Path,
    autogrid_cmd: str = "autogrid4",
) -> bool:
    """
    调用 autogrid4 生成 .map 网格地图文件

    Args:
        gpf_file: GPF 文件路径
        log_file: 日志文件路径 (.glg)
        work_dir: 工作目录（autogrid4 会在该目录下生成 .map 文件）
        autogrid_cmd: autogrid4 可执行文件路径

    Returns:
        是否成功
    """
    gpf_file = Path(gpf_file).resolve()
    log_file = Path(log_file).resolve()
    work_dir = Path(work_dir).resolve()

    cmd = [autogrid_cmd, "-p", str(gpf_file), "-l", str(log_file)]
    logger.info(f"运行 AutoGrid4: {' '.join(cmd)}")
    logger.info(f"工作目录: {work_dir}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=600,  # 10 分钟超时
        )

        if result.returncode == 0:
            logger.info(f"AutoGrid4 成功完成！日志文件: {log_file}")
            return True
        else:
            # 检测段错误 (signal 11 = SIGSEGV, exit code 128+11=139)
            if result.returncode == -11 or result.returncode == 139:
                logger.error(
                    f"AutoGrid4 段错误 (exit code {result.returncode})。\n"
                    f"这是 autogrid4 4.2.6 在较新 Linux 系统上的已知问题。\n"
                    f"建议: 减小网格大小或使用较旧的 Linux 兼容版本。"
                )
            else:
                stderr_msg = (result.stderr or "")[:1000]
                stdout_msg = (result.stdout or "")[:1000]
                logger.error(
                    f"AutoGrid4 失败 (exit code {result.returncode}):\n"
                    f"stderr: {stderr_msg}\n"
                    f"stdout: {stdout_msg}"
                )
            return False

    except subprocess.TimeoutExpired:
        logger.error("AutoGrid4 超时（超过 600 秒）")
        return False
    except FileNotFoundError:
        logger.error(f"找不到 autogrid4 可执行文件: {autogrid_cmd}")
        return False
    except Exception as e:
        logger.error(f"AutoGrid4 运行出错: {e}")
        return False


def collect_map_files(work_dir: Path, receptor_stem: str) -> List[Path]:
    """
    收集生成的 .map 文件

    Args:
        work_dir: 工作目录
        receptor_stem: 受体文件名（不含扩展名）

    Returns:
        .map 文件路径列表
    """
    map_files = sorted(Path(work_dir).glob(f"{receptor_stem}.*.map"))
    return map_files


def main():
    parser = argparse.ArgumentParser(
        description="AutoGrid4 网格地图生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 方式1: 直接指定网格参数
  python run_autogrid.py \\
      --receptor ../output/prepared_protein/4dkl.pdbqt \\
      --ligand ../output/rounds/round_001/03_pdbqt/ligand_0.pdbqt \\
      --center_x -27.997 --center_y -13.204 --center_z -11.084 \\
      --size_x 17.642 --size_y 23.867 --size_z 20.299

  # 方式2: 从参数文件读取
  python run_autogrid.py \\
      --receptor ../output/prepared_protein/4dkl.pdbqt \\
      --ligand ../output/rounds/round_001/03_pdbqt/ligand_0.pdbqt \\
      --param_file ../output/grid/4dkl.txt

  # 方式3: 仅生成 GPF 文件（不运行 autogrid4）
  python run_autogrid.py \\
      --receptor ../output/prepared_protein/4dkl.pdbqt \\
      --ligand ../output/rounds/round_001/03_pdbqt/ligand_0.pdbqt \\
      --param_file ../output/grid/4dkl.txt \\
      --gpf_only
        """,
    )

    # 输入文件参数
    parser.add_argument(
        "--receptor", "-r",
        type=str,
        required=True,
        help="受体 PDBQT 文件路径",
    )
    parser.add_argument(
        "--ligand", "-l",
        type=str,
        default=None,
        help="配体 PDBQT 文件路径（用于自动检测原子类型）",
    )

    # 网格参数（方式1: 直接指定）
    parser.add_argument("--center_x", type=float, default=None, help="网格中心 X 坐标")
    parser.add_argument("--center_y", type=float, default=None, help="网格中心 Y 坐标")
    parser.add_argument("--center_z", type=float, default=None, help="网格中心 Z 坐标")
    parser.add_argument("--size_x", type=float, default=None, help="网格 X 方向大小 (Å)")
    parser.add_argument("--size_y", type=float, default=None, help="网格 Y 方向大小 (Å)")
    parser.add_argument("--size_z", type=float, default=None, help="网格 Z 方向大小 (Å)")

    # 网格参数（方式2: 从参数文件读取）
    parser.add_argument(
        "--param_file", "-p",
        type=str,
        default=None,
        help="口袋参数文件路径 (.txt)，包含 center_x/y/z 和 size_x/y/z",
    )

    # 输出参数
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="输出目录（默认: 当前目录下的 autogrid_output）",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="输出文件前缀（默认: 受体文件名）",
    )

    # 运行参数
    parser.add_argument(
        "--spacing",
        type=float,
        default=DEFAULT_SPACING,
        help=f"网格间距，单位 Å（默认: {DEFAULT_SPACING}）",
    )
    parser.add_argument(
        "--gpf_only",
        action="store_true",
        help="仅生成 GPF 文件，不运行 autogrid4",
    )
    parser.add_argument(
        "--autogrid_cmd",
        type=str,
        default=None,
        help="autogrid4 可执行文件路径（默认: 自动查找）",
    )
    parser.add_argument(
        "--ligand_types",
        type=str,
        nargs="+",
        default=None,
        help=f"原子类型列表（默认: {' '.join(DEFAULT_LIGAND_TYPES)}）",
    )

    args = parser.parse_args()

    # ============================================================
    # 1. 检查输入文件
    # ============================================================
    receptor_pdbqt = Path(args.receptor)
    if not receptor_pdbqt.exists():
        logger.error(f"受体文件不存在: {receptor_pdbqt}")
        sys.exit(1)

    ligand_pdbqt = Path(args.ligand) if args.ligand else None
    if ligand_pdbqt and not ligand_pdbqt.exists():
        logger.warning(f"配体文件不存在: {ligand_pdbqt}，将使用默认原子类型")
        ligand_pdbqt = None

    # ============================================================
    # 2. 获取网格参数
    # ============================================================
    grid_params = None

    if args.param_file:
        # 方式2: 从参数文件读取
        param_file = Path(args.param_file)
        if not param_file.exists():
            logger.error(f"参数文件不存在: {param_file}")
            sys.exit(1)
        grid_params = parse_param_file(param_file)
    elif all(
        v is not None
        for v in [args.center_x, args.center_y, args.center_z, args.size_x, args.size_y, args.size_z]
    ):
        # 方式1: 直接指定
        grid_params = {
            "center_x": args.center_x,
            "center_y": args.center_y,
            "center_z": args.center_z,
            "size_x": args.size_x,
            "size_y": args.size_y,
            "size_z": args.size_z,
        }
    else:
        logger.error(
            "请提供网格参数：\n"
            "  方式1: 使用 --center_x/y/z 和 --size_x/y/z 直接指定\n"
            "  方式2: 使用 --param_file 从参数文件读取"
        )
        sys.exit(1)

    # ============================================================
    # 3. 设置输出目录
    # ============================================================
    output_dir = Path(args.output_dir) if args.output_dir else Path("autogrid_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定输出文件前缀
    prefix = args.prefix or receptor_pdbqt.stem

    # ============================================================
    # 4. 复制受体到工作目录（autogrid4 需要）
    # ============================================================
    dest_receptor = output_dir / receptor_pdbqt.name
    if not dest_receptor.exists():
        shutil.copy2(str(receptor_pdbqt), str(dest_receptor))
        logger.info(f"受体已复制到工作目录: {dest_receptor}")

    # 如果有配体，也复制到工作目录
    dest_ligand = None
    if ligand_pdbqt:
        dest_ligand = output_dir / ligand_pdbqt.name
        if not dest_ligand.exists():
            shutil.copy2(str(ligand_pdbqt), str(dest_ligand))
            logger.info(f"配体已复制到工作目录: {dest_ligand}")

    # ============================================================
    # 5. 生成 GPF 文件
    # ============================================================
    gpf_file = output_dir / f"{prefix}.gpf"
    logger.info("=" * 60)
    logger.info("步骤1: 生成 GPF 参数文件")
    logger.info("=" * 60)

    generate_gpf_with_adt(
        receptor_pdbqt=dest_receptor,
        ligand_pdbqt=dest_ligand,
        grid_params=grid_params,
        output_gpf=gpf_file,
        spacing=args.spacing,
        ligand_types=args.ligand_types,
    )

    # ============================================================
    # 6. 运行 AutoGrid4
    # ============================================================
    if args.gpf_only:
        logger.info("--gpf_only 模式，跳过 AutoGrid4 运行")
        logger.info(f"GPF 文件: {gpf_file}")
        logger.info("你可以稍后手动运行:")
        logger.info(f"  autogrid4 -p {gpf_file} -l {output_dir}/{prefix}.glg")
        return

    logger.info("=" * 60)
    logger.info("步骤2: 运行 AutoGrid4 生成 .map 文件")
    logger.info("=" * 60)

    # 查找 autogrid4
    autogrid_cmd = args.autogrid_cmd or find_autogrid4()
    if not autogrid_cmd:
        logger.error(
            "找不到 autogrid4 可执行文件！\n"
            "请确保已安装 AutoGrid4，或使用 --autogrid_cmd 指定路径。\n"
            "安装方法: sudo apt install autogrid  (Ubuntu/Debian)\n"
            "        或从 https://ccsb.scripps.edu/adfr/downloads/ 下载"
        )
        sys.exit(1)

    logger.info(f"使用 AutoGrid4: {autogrid_cmd}")

    # 运行 autogrid4
    log_file = output_dir / f"{prefix}.glg"
    success = run_autogrid4(
        gpf_file=gpf_file,
        log_file=log_file,
        work_dir=output_dir,
        autogrid_cmd=autogrid_cmd,
    )

    # ============================================================
    # 7. 收集结果
    # ============================================================
    logger.info("=" * 60)
    logger.info("结果汇总")
    logger.info("=" * 60)

    if success:
        map_files = collect_map_files(output_dir, receptor_pdbqt.stem)
        logger.info(f"✅ AutoGrid4 运行成功！")
        logger.info(f"   GPF 文件: {gpf_file}")
        logger.info(f"   GLG 日志: {log_file}")
        logger.info(f"   生成的 .map 文件 ({len(map_files)} 个):")
        for mf in map_files:
            size_kb = os.path.getsize(mf) / 1024
            logger.info(f"     - {mf.name} ({size_kb:.1f} KB)")
    else:
        logger.error(f"❌ AutoGrid4 运行失败！")
        logger.info(f"   GPF 文件已生成: {gpf_file}")
        logger.info(f"   请检查日志: {log_file}")
        logger.info("可能的原因:")
        logger.info("  1. 网格过大（尝试减小 size_x/y/z）")
        logger.info("  2. autogrid4 版本与系统不兼容")
        logger.info("  3. 受体 PDBQT 文件格式问题")
        sys.exit(1)


if __name__ == "__main__":
    main()

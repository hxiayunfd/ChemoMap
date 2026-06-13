#!/usr/bin/env python3
"""
模块: 计算化学报告生成器
Module: Computational Chemistry Report Generator

每轮迭代结束后自动生成:
1. 对接分数分布图 (binding energy histogram)
2. 理化性质漂移分析 (MW, cLogP, TPSA, Fsp3, RB KDE)
3. Bemis-Murcko 骨架富集分析 (Fisher's exact test)
4. 化学空间 t-SNE 拓扑映射
5. PDF 综合报告 + Excel 数据表
6. Top-N 优先命中分子的结构卡片

用法:
    from report_generator import generate_round_report

    generate_round_report(
        docking_results_df=df,
        round_num=1,
        output_dir=Path("output/rounds/round_001/06_report"),
    )
"""

import io
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 无头环境兼容
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ── RDKit 惰性导入 ──
_rdkit_available = False
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw, AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    _rdkit_available = True
except ImportError:
    pass

# ── reportlab 惰性导入 ──
_reportlab_available = False
_canvas = None
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether, Image,
    )
    from reportlab.pdfgen import canvas as _canvas
    _reportlab_available = True
except ImportError:
    pass

# ── 绘图配置 ──
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="ticks", context="talk")


# ============================================================
# PDF 画布 (工业级页眉页脚)
# ============================================================

NumberedCanvas = None
if _canvas is not None:
    class NumberedCanvas(_canvas.Canvas):
        """自动绘制页眉页脚 + 动态总页数"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self._draw_header_footer(num_pages)
                super().showPage()
            super().save()

        def _draw_header_footer(self, page_count):
            self.saveState()
            if self._pageNumber > 1:
                self.setFont("Helvetica", 9)
                self.setFillColor(colors.HexColor("#666666"))
                self.drawString(54, 750, "Molecular Directed Evolution — Round Report")
                self.setStrokeColor(colors.HexColor("#e1e4e8"))
                self.setLineWidth(0.5)
                self.line(54, 742, 558, 742)
                self.drawRightString(558, 40, f"Page {self._pageNumber} of {page_count}")
                self.drawString(54, 40, "CONFIDENTIAL — MedChem Decision Document")
                self.line(54, 52, 558, 52)
            self.restoreState()


# ============================================================
# 理化性质计算
# ============================================================

def _safe_property(mol, func, default=np.nan):
    """安全计算分子描述符"""
    try:
        return func(mol)
    except Exception:
        return default


def calculate_cheminformatics_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算分子描述符 + Bemis-Murcko 骨架

    新增列: MW, cLogP, TPSA, HBA, HBD, RB, Fsp3, Scaffold
    """
    if not _rdkit_available:
        logger.warning("RDKit not available, skipping property calculation")
        return df

    logger.info("Calculating cheminformatics properties...")

    props = {
        "MW": [], "cLogP": [], "TPSA": [], "HBA": [], "HBD": [],
        "RB": [], "Fsp3": [], "Scaffold": [],
    }

    for smi in df.get("SMILES", []):
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            for k in props:
                props[k].append(np.nan)
            continue

        props["MW"].append(_safe_property(mol, Descriptors.MolWt))
        props["cLogP"].append(_safe_property(mol, Descriptors.MolLogP))
        props["TPSA"].append(_safe_property(mol, Descriptors.TPSA))
        props["HBA"].append(_safe_property(mol, Descriptors.NumHAcceptors))
        props["HBD"].append(_safe_property(mol, Descriptors.NumHDonors))
        props["RB"].append(_safe_property(mol, Descriptors.NumRotatableBonds))
        props["Fsp3"].append(_safe_property(mol, Descriptors.FractionCSP3))

        # Bemis-Murcko 骨架
        try:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(core) if core else "Acyclic"
        except Exception:
            scaffold_smi = "Analysis_Failed"
        props["Scaffold"].append(scaffold_smi)

    for k, v in props.items():
        df[k] = v
    return df


# ============================================================
# 图表生成
# ============================================================

def generate_score_distribution(
    df: pd.DataFrame, output_dir: Path, energy_col: str = "target_energy",
):
    """对接分数分布图（自动检测单/双靶点）"""
    valid = df[df.get("docking_success", True) == True]
    if valid.empty:
        return

    has_anti = "anti_energy" in valid.columns and valid["anti_energy"].notna().any()

    if has_anti:
        # 双靶点：三面板
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        cols = ["target_energy", "anti_energy", "delta"]
        titles = ["Target Score", "Anti-Target Score", "Selectivity (Delta)"]
        colors = ["#2b5c8f", "#d95f02", "#7570b3"]

        for i, (col, title, color) in enumerate(zip(cols, titles, colors)):
            energies = valid[col].dropna()
            if energies.empty:
                continue
            sns.histplot(energies, kde=True, ax=axes[i], color=color, bins=30)
            axes[i].axvline(energies.mean(), color="red", linestyle="--", alpha=0.7,
                            label=f"μ={energies.mean():.2f}")
            axes[i].set_title(f"{title}\n(μ={energies.mean():.2f}, σ={energies.std():.2f})")
            axes[i].set_xlabel("kcal/mol")
            axes[i].legend(fontsize=9)
    else:
        # 单靶点
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        if energy_col not in valid.columns:
            for alt in ["target_energy", "binding_energy"]:
                if alt in valid.columns:
                    energy_col = alt
                    break
        energies = valid[energy_col].dropna()

        sns.histplot(energies, kde=True, ax=axes[0], color="#2b5c8f", bins=30)
        axes[0].axvline(energies.mean(), color="red", linestyle="--", alpha=0.7,
                        label=f"Mean: {energies.mean():.2f}")
        p10 = np.percentile(energies, 10)
        axes[0].axvline(p10, color="green", linestyle="--", alpha=0.7,
                        label=f"Top 10%: {p10:.2f}")
        axes[0].set_title(f"Binding Energy Distribution\n"
                          f"(μ={energies.mean():.2f}, σ={energies.std():.2f} kcal/mol)")
        axes[0].set_xlabel("kcal/mol")
        axes[0].legend()

        sorted_e = np.sort(energies)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        axes[1].plot(sorted_e, cdf, color="#2b5c8f", linewidth=2)
        axes[1].axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        axes[1].axvline(energies.median(), color="gray", linestyle=":", alpha=0.5)
        axes[1].set_title("Cumulative Distribution")
        axes[1].set_xlabel("kcal/mol")
        axes[1].set_ylabel("Cumulative Fraction")

    plt.tight_layout()
    path = output_dir / "binding_energy_distribution.png"
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"  Score distribution → {path}")

    # 双靶点额外：Pareto 前沿
    if has_anti:
        _generate_pareto_front(valid, output_dir)


def _generate_pareto_front(df: pd.DataFrame, output_dir: Path):
    """双靶点 Pareto 前沿：目标亲和力 vs 选择性"""
    valid = df.dropna(subset=["target_energy", "anti_energy"]).copy()
    if len(valid) < 5:
        return

    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(
        valid["target_energy"], valid["anti_energy"],
        c=valid["delta"], cmap="coolwarm_r", s=20, alpha=0.6,
    )
    plt.colorbar(scatter, label="Delta (selectivity)")

    # Pareto 前沿
    pareto = valid[valid["target_energy"] < valid["target_energy"].quantile(0.5)]
    if not pareto.empty:
        sorted_pf = pareto.sort_values("target_energy")
        front = []
        best_anti = float("inf")
        for _, row in sorted_pf.iterrows():
            if row["anti_energy"] < best_anti:
                front.append(row)
                best_anti = row["anti_energy"]
        if front:
            pf = pd.DataFrame(front)
            plt.plot(pf["target_energy"], pf["anti_energy"], "r-o", linewidth=2, markersize=6, label="Pareto Front")

    plt.axhline(0, color="gray", linestyle="--", alpha=0.5, label="No binding (anti)")
    plt.xlabel("Target Binding Energy (kcal/mol)")
    plt.ylabel("Anti-Target Binding Energy (kcal/mol)")
    plt.title("Target vs Anti-Target Affinity Landscape")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "pareto_front.png"
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"  Pareto front → {path}")


def generate_property_drift(
    df: pd.DataFrame, output_dir: Path, energy_col: str = "binding_energy",
):
    """
    理化性质漂移分析

    比较全体分子 vs Top 10% vs Top 1% 的物性分布变化，
    揭示定向进化是否在隐含地筛选某种物性模式。
    """
    valid = df.dropna(subset=[energy_col]).copy()
    if valid.empty:
        return

    # 检查是否有物性列
    prop_cols = ["MW", "cLogP", "TPSA", "Fsp3", "RB"]
    available = [c for c in prop_cols if c in valid.columns]
    if not available:
        return

    # 分级切分
    top10_cutoff = valid[energy_col].quantile(0.10)
    top1_cutoff = valid[energy_col].quantile(0.01)

    df_top10 = valid[valid[energy_col] <= top10_cutoff]
    df_top1 = valid[valid[energy_col] <= top1_cutoff]

    n_cols = min(3, len(available))
    n_rows = int(np.ceil(len(available) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(available):
        ax = axes[i]
        col_data = valid[col].dropna()
        # 跳过零方差的列（例如所有分子都有相同的属性值）
        if col_data.nunique() < 2:
            ax.text(0.5, 0.5, f"{col}\n(insufficient variance for KDE)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{col} Property Drift")
            continue

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Dataset has 0 variance")
            sns.kdeplot(col_data, ax=ax, label="Full Library",
                        color="#718096", linewidth=2, fill=True, alpha=0.05)
            sns.kdeplot(df_top10[col].dropna(), ax=ax, label="Top 10%",
                        color="#2b5c8f", linewidth=2.5)
            sns.kdeplot(df_top1[col].dropna(), ax=ax, label="Top 1% Elite",
                        color="#e53e3e", linewidth=3)
        ax.set_title(f"{col} Property Drift")
        ax.set_xlabel(col)
        ax.legend(fontsize=9)

    # 隐藏多余子图
    for i in range(len(available), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    path = output_dir / "property_drift.png"
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"  Property drift → {path}")


def generate_chemical_space_tsne(
    df: pd.DataFrame, output_dir: Path, energy_col: str = "binding_energy",
):
    """
    t-SNE 化学空间映射

    用 Morgan 指纹 + PCA 降维 + t-SNE 可视化分子在化学空间中的分布，
    标注高分分子（Top 10%）的位置。
    """
    if not _rdkit_available or "SMILES" not in df.columns:
        return

    valid = df.dropna(subset=[energy_col]).copy()
    if len(valid) < 20:
        return

    logger.info("  Computing Morgan fingerprints for t-SNE...")

    # 提取指纹
    try:
        from rdkit.Chem import rdFingerprintGenerator
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        use_modern = True
    except Exception:
        use_modern = False

    fps, valid_idx = [], []
    for i, smi in enumerate(valid["SMILES"]):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        if use_modern:
            fps.append(fp_gen.GetFingerprintAsNumPy(mol))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            arr = np.zeros((1,))
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        valid_idx.append(i)

    if len(fps) < 10:
        return

    fps_matrix = np.array(fps)

    # PCA 预降维
    pca_dim = min(50, fps_matrix.shape[0], fps_matrix.shape[1])
    X_pca = PCA(n_components=pca_dim, random_state=42).fit_transform(fps_matrix)

    # t-SNE
    from sklearn.manifold import TSNE
    try:
        embedding = TSNE(
            n_components=2, perplexity=min(30, len(fps) - 1),
            random_state=42, max_iter=1000, n_jobs=-1,
        ).fit_transform(X_pca)
    except TypeError:
        embedding = TSNE(
            n_components=2, perplexity=min(30, len(fps) - 1),
            random_state=42, n_iter=1000, n_jobs=-1,
        ).fit_transform(X_pca)

    # 绘图
    df_tsne = valid.iloc[valid_idx].copy()
    df_tsne["Dim_1"] = embedding[:, 0]
    df_tsne["Dim_2"] = embedding[:, 1]

    top10 = valid[energy_col].quantile(0.10)
    df_tsne["is_hit"] = df_tsne[energy_col] <= top10

    plt.figure(figsize=(10, 8))
    plt.scatter(
        df_tsne[~df_tsne["is_hit"]]["Dim_1"],
        df_tsne[~df_tsne["is_hit"]]["Dim_2"],
        c="#e2e8f0", s=15, alpha=0.6, label="General Pool",
    )
    df_hit = df_tsne[df_tsne["is_hit"]]
    sc = plt.scatter(
        df_hit["Dim_1"], df_hit["Dim_2"],
        c=df_hit[energy_col], cmap="YlOrRd_r", s=35, alpha=0.9,
        label=f"Top 10% (≤{top10:.2f} kcal/mol)",
    )
    plt.colorbar(sc, label="Binding Energy (kcal/mol)")
    plt.title("Chemical Space Topology (t-SNE / Morgan Fingerprint)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    path = output_dir / "chemical_space_tsne.png"
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"  Chemical space → {path}")


def generate_scaffold_enrichment(
    df: pd.DataFrame, output_dir: Path, energy_col: str = "binding_energy",
) -> Optional[pd.DataFrame]:
    """
    Bemis-Murcko 骨架富集分析

    对每个骨架做 Fisher exact test，看是否在选择性子集中显著富集。
    "选择性"在此定义为 Top 10% 结合能分子。
    """
    if "Scaffold" not in df.columns:
        return None

    valid = df.dropna(subset=[energy_col]).copy()
    if valid.empty:
        return None

    top10 = valid[energy_col].quantile(0.10)
    valid["is_hit"] = valid[energy_col] <= top10

    n_total = len(valid)
    n_hit_total = valid["is_hit"].sum()

    if n_hit_total < 3:
        return None

    from scipy.stats import fisher_exact

    scaffold_stats = valid.groupby("Scaffold").agg(
        n_hit=("is_hit", "sum"),
        n_total=("is_hit", "count"),
    ).reset_index()

    ef_list, or_list, p_list = [], [], []
    for _, row in scaffold_stats.iterrows():
        a = row["n_hit"]
        b = row["n_total"] - a
        c = n_hit_total - a
        d = (n_total - n_hit_total) - b

        ef = (a / row["n_total"]) / (n_hit_total / n_total) if row["n_total"] > 0 else 0
        ef_list.append(ef)

        table = np.array([[a, b], [c, d]])
        oddsr, pval = fisher_exact(table, alternative="greater")
        or_list.append(oddsr)
        p_list.append(pval)

    scaffold_stats["Enrichment_Factor"] = ef_list
    scaffold_stats["Odds_Ratio"] = or_list
    scaffold_stats["Fisher_Pvalue"] = p_list

    # 过滤：至少 3 个分子
    enriched = scaffold_stats[scaffold_stats["n_total"] >= 3].sort_values(
        "n_hit", ascending=False
    ).head(15)

    enriched.to_csv(output_dir / "scaffold_enrichment.csv", index=False)

    # 渲染 Top 10 骨架图片矩阵
    _render_scaffold_grid(enriched.head(10), output_dir)

    return enriched


def _render_scaffold_grid(enriched_df: pd.DataFrame, output_dir: Path):
    """渲染骨架 2D 结构矩阵"""
    if not _rdkit_available or enriched_df.empty:
        return

    mols, legends = [], []
    for i, (_, row) in enumerate(enriched_df.iterrows()):
        scaffold_smi = row["Scaffold"]
        mol = Chem.MolFromSmiles("CC") if scaffold_smi == "Acyclic" else Chem.MolFromSmiles(scaffold_smi)
        if mol is not None:
            Chem.rdDepictor.Compute2DCoords(mol)
            mols.append(mol)
            legends.append(
                f"EF:{row['Enrichment_Factor']:.1f}x "
                f"OR:{row['Odds_Ratio']:.1f} "
                f"p:{row['Fisher_Pvalue']:.2e}"
            )

    if not mols:
        return

    try:
        img = Draw.MolsToGridImage(
            mols, molsPerRow=5, subImgSize=(240, 240),
            legends=legends, useSVG=False,
        )
        path = output_dir / "top_scaffolds.png"
        if hasattr(img, "save"):
            img.save(str(path))
        else:
            with open(path, "wb") as f:
                f.write(img)
        logger.info(f"  Scaffold matrix → {path}")
    except Exception as e:
        logger.warning(f"  Scaffold rendering failed: {e}")


# ============================================================
# PDF 报告
# ============================================================

def _fmt(val):
    """格式化数值，None/NaN → 'N/A'"""
    if val is None:
        return "N/A"
    try:
        if np.isnan(float(val)):
            return "N/A"
        return f"{float(val):.2f}"
    except (ValueError, TypeError):
        return str(val)


def _render_molecule_card(row, styles) -> Optional:
    """渲染单个分子的信息卡片（PDF 用）"""
    if not _rdkit_available or not _reportlab_available:
        return None

    smi = str(row.get("SMILES", ""))
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    # 2D 结构图
    Chem.rdDepictor.Compute2DCoords(mol)
    try:
        img_pil = Draw.MolToImage(mol, size=(600, 600))
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        buf.seek(0)
        img = Image(buf, width=165, height=165)
    except Exception:
        return None

    # 双靶点列
    target_e = _fmt(row.get("target_energy"))
    anti_e = _fmt(row.get("anti_energy"))
    delta_e = _fmt(row.get("delta"))
    compound_id = str(row.get("Compound", row.get("SMILES", "?")))[:40]

    # 右侧文本
    right_flowables = [
        Paragraph(f"ID: {compound_id}",
                  ParagraphStyle("CardID", fontName="Helvetica-Bold", fontSize=11,
                                 leading=13, textColor=colors.HexColor("#1a1a1a"))),
        Spacer(1, 4),
        Paragraph("Docking Results:",
                  ParagraphStyle("SectBlue", fontName="Helvetica-Bold", fontSize=10,
                                 leading=12, textColor=colors.HexColor("#2b5c8f"))),
        Paragraph(f"• NMDA Affinity: <b>{target_e}</b> kcal/mol",
                  ParagraphStyle("CardItem", fontName="Helvetica", fontSize=9,
                                 leading=12, textColor=colors.HexColor("#333333"))),
        Paragraph(f"• Opioid Affinity: {anti_e} kcal/mol",
                  ParagraphStyle("CardItem", fontName="Helvetica", fontSize=9,
                                 leading=12, textColor=colors.HexColor("#333333"))),
        Paragraph(f"• Delta (Δ): {delta_e} kcal/mol",
                  ParagraphStyle("CardItemD", fontName="Helvetica-Bold" if row.get("delta", 0) < 0 else "Helvetica",
                                 fontSize=9, leading=12,
                                 textColor=colors.HexColor("#d95f02" if row.get("delta", 0) < 0 else "#333333"))),
        Spacer(1, 2),
        Paragraph("Physicochemical Properties:",
                  ParagraphStyle("SectGreen", fontName="Helvetica-Bold", fontSize=10,
                                 leading=12, textColor=colors.HexColor("#1b9e77"))),
        Paragraph(
            f"• MW: {row.get('MW', '?'):.1f} Da  |  "
            f"cLogP: {row.get('cLogP', '?'):.2f}  |  "
            f"TPSA: {row.get('TPSA', '?'):.1f} Å²",
            ParagraphStyle("CardItem2", fontName="Helvetica", fontSize=9,
                           leading=11, textColor=colors.HexColor("#333333")),
        ),
        Paragraph(
            f"• Fsp3: {row.get('Fsp3', '?'):.2f}  |  "
            f"HBA/HBD: {int(row.get('HBA', 0))}/{int(row.get('HBD', 0))}  |  "
            f"RotBonds: {int(row.get('RB', 0))}",
            ParagraphStyle("CardItem2", fontName="Helvetica", fontSize=9,
                           leading=11, textColor=colors.HexColor("#333333")),
        ),
    ]

    card = Table([[img, right_flowables]], colWidths=[175, 305])
    card.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (1, 0), (1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))

    outer = Table([[card]], colWidths=[494])
    outer.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e1e4e8")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafafa")),
        ("PADDING", (0, 0), (-1, -1), 2),
    ]))
    return outer


def generate_pdf_report(
    df: pd.DataFrame,
    top_hits: pd.DataFrame,
    scaffold_df: Optional[pd.DataFrame],
    output_dir: Path,
    round_num: int,
    summary: dict,
):
    """生成综合 PDF 报告"""
    if not _reportlab_available:
        logger.warning("reportlab not available, skipping PDF generation")
        return

    pdf_path = output_dir / f"round_{round_num:03d}_report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=letter,
        leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54,
    )

    # 样式
    title_style = ParagraphStyle(
        "Title", fontName="Helvetica-Bold", fontSize=23, leading=28,
        textColor=colors.HexColor("#1a365d"), spaceAfter=15,
    )
    h1_style = ParagraphStyle(
        "H1", fontName="Helvetica-Bold", fontSize=15, leading=19,
        textColor=colors.HexColor("#1a365d"), spaceBefore=16, spaceAfter=10,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body", fontName="Helvetica", fontSize=10, leading=14,
        textColor=colors.HexColor("#333333"), spaceAfter=8,
    )

    story = []

    # ── 封面 ──
    story.append(Spacer(1, 40))
    story.append(Table(
        [[""]], colWidths=[504], rowHeights=[6],
        style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1a365d"))]),
    ))
    story.append(Spacer(1, 25))
    story.append(Paragraph(
        "MOLECULAR DIRECTED EVOLUTION CAMPAIGN",
        ParagraphStyle("Sub", fontName="Helvetica-Bold", fontSize=11,
                       textColor=colors.HexColor("#4a5568"), spaceAfter=10),
    ))
    story.append(Paragraph(
        f"Round {round_num} — Docking & Cheminformatics Report", title_style,
    ))
    story.append(Paragraph(
        "AutoDock-GPU Virtual Screening + Reinvent4 Generative Chemistry",
        body_style,
    ))
    story.append(Spacer(1, 200))

    # 摘要元信息
    best_e = summary.get("best_binding_energy", "N/A")
    meta = (
        f"<b>Compounds Generated:</b> {summary.get('generated', 'N/A')}<br/>"
        f"<b>After Property Filter:</b> {summary.get('after_filter', 'N/A')}<br/>"
        f"<b>Successfully Docked:</b> {summary.get('docking_success', 'N/A')}<br/>"
        f"<b>Best Binding Energy:</b> {best_e:.2f} kcal/mol" if isinstance(best_e, (int, float)) else f"<b>Best Binding Energy:</b> {best_e}<br/>"
        f"<b>Seeds Selected:</b> {summary.get('seeds_selected', 'N/A')}<br/>"
        f"<b>Timestamp:</b> {summary.get('timestamp', 'N/A')}"
    )
    story.append(Paragraph(meta, ParagraphStyle(
        "Meta", fontName="Helvetica", fontSize=9.5, leading=15,
        textColor=colors.HexColor("#718096"),
    )))
    story.append(PageBreak())

    # ── Section 1: 对接分数分布 ──
    story.append(Paragraph("1. Binding Energy Distribution", h1_style))
    story.append(Paragraph(
        "Distribution of AutoDock-GPU predicted binding free energies. "
        "More negative values indicate stronger predicted binding. "
        "The red dashed line marks the mean, and the green line marks the top 10% threshold.",
        body_style,
    ))
    png = output_dir / "binding_energy_distribution.png"
    if png.exists():
        story.append(KeepTogether([
            Image(str(png), width=490, height=175),
            Paragraph("<b>Figure 1:</b> Binding energy histogram and cumulative distribution.",
                      ParagraphStyle("Cap", fontName="Helvetica-Oblique", fontSize=8.5,
                                     alignment=1, spaceBefore=4)),
        ]))
    story.append(PageBreak())

    # ── Section 2: 骨架富集 ──
    story.append(Paragraph("2. Scaffold Enrichment Analysis", h1_style))
    story.append(Paragraph(
        "Bemis-Murcko molecular scaffolds enriched among top-scoring compounds. "
        "Enrichment Factor (EF) > 1 indicates over-representation in the hit set. "
        "Fisher's exact test p-values assess statistical significance.",
        body_style,
    ))
    if scaffold_df is not None and not scaffold_df.empty:
        story.append(_build_scaffold_table(scaffold_df.head(10)))
    scaffold_png = output_dir / "top_scaffolds.png"
    if scaffold_png.exists():
        story.append(KeepTogether([
            Image(str(scaffold_png), width=490, height=196),
            Paragraph("<b>Figure 2:</b> Top enriched Bemis-Murcko scaffolds (2D structures).",
                      ParagraphStyle("Cap", fontName="Helvetica-Oblique", fontSize=8.5,
                                     alignment=1, spaceBefore=4)),
        ]))
    story.append(PageBreak())

    # ── Section 3: 物性漂移 ──
    story.append(Paragraph("3. Property Drift Analysis", h1_style))
    story.append(Paragraph(
        "KDE comparison of key physicochemical properties across three tiers: "
        "full library (gray), top 10% (blue), and top 1% elite (red). "
        "Systematic shifts reveal implicit property selection during molecular evolution.",
        body_style,
    ))
    drift_png = output_dir / "property_drift.png"
    if drift_png.exists():
        story.append(KeepTogether([
            Image(str(drift_png), width=490, height=270),
            Paragraph("<b>Figure 3:</b> Multi-stage property drift across selection tiers.",
                      ParagraphStyle("Cap", fontName="Helvetica-Oblique", fontSize=8.5,
                                     alignment=1, spaceBefore=4)),
        ]))
    story.append(PageBreak())

    # ── Section 4: 化学空间 ──
    story.append(Paragraph("4. Chemical Space Topology (t-SNE)", h1_style))
    story.append(Paragraph(
        "t-SNE projection of Morgan fingerprint space. "
        "Colored points indicate top 10% scoring compounds. "
        "Clustering of hits in specific regions suggests privileged chemotypes.",
        body_style,
    ))
    tsne_png = output_dir / "chemical_space_tsne.png"
    if tsne_png.exists():
        story.append(KeepTogether([
            Image(str(tsne_png), width=430, height=344),
            Paragraph("<b>Figure 4:</b> Chemical space convergence via t-SNE topology.",
                      ParagraphStyle("Cap", fontName="Helvetica-Oblique", fontSize=8.5,
                                     alignment=1, spaceBefore=4)),
        ]))
    story.append(PageBreak())

    # ── Section 5: Top Hits ──
    story.append(Paragraph("5. Prioritized Hit Candidates (Top 50)", h1_style))
    story.append(Spacer(1, 5))
    styles = getSampleStyleSheet()
    for _, row in top_hits.head(50).iterrows():
        card = _render_molecule_card(row, styles)
        if card:
            story.append(KeepTogether([card, Spacer(1, 8)]))

    doc.build(story, canvasmaker=NumberedCanvas)
    logger.info(f"  PDF report → {pdf_path}")


def _build_scaffold_table(scaffold_df: pd.DataFrame):
    """构建骨架富集 PDF 表格"""
    head_style = ParagraphStyle(
        "THead", fontName="Helvetica-Bold", fontSize=9,
        leading=11, alignment=1, textColor=colors.white,
    )
    text_style = ParagraphStyle(
        "TText", fontName="Helvetica", fontSize=8.5, leading=11, alignment=1,
    )

    table_data = [[
        Paragraph("Scaffold SMILES", head_style),
        Paragraph("N Hit", head_style),
        Paragraph("N Total", head_style),
        Paragraph("EF", head_style),
        Paragraph("Odds Ratio", head_style),
        Paragraph("Fisher p", head_style),
    ]]
    for _, row in scaffold_df.iterrows():
        smi = row["Scaffold"]
        smi_short = (smi[:25] + "...") if len(str(smi)) > 28 else str(smi)
        table_data.append([
            Paragraph(f"<code>{smi_short}</code>",
                      ParagraphStyle("LC", fontName="Courier", fontSize=8, leading=10)),
            Paragraph(str(int(row["n_hit"])), text_style),
            Paragraph(str(int(row["n_total"])), text_style),
            Paragraph(f"{row['Enrichment_Factor']:.2f}x", text_style),
            Paragraph(f"{row['Odds_Ratio']:.2f}", text_style),
            Paragraph(f"{row['Fisher_Pvalue']:.2e}", text_style),
        ])

    tbl = Table(table_data, colWidths=[184, 55, 55, 60, 70, 80])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a365d")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


# ============================================================
# 主入口
# ============================================================

def generate_round_report(
    docking_results_df: pd.DataFrame,
    round_num: int,
    output_dir: Path,
    summary: Optional[dict] = None,
    energy_col: str = "target_energy",
    top_n: int = 50,
) -> Path:
    """生成单轮完整计算化学报告（自动检测单/双靶点）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summary or {}

    logger.info(f"Generating Round {round_num} report → {output_dir}")

    # 检测双靶点模式
    has_anti = ("anti_energy" in docking_results_df.columns and
                docking_results_df["anti_energy"].notna().any())

    # 确定排序依据
    if has_anti and "delta" in docking_results_df.columns:
        sort_col = "delta"
    elif energy_col in docking_results_df.columns:
        sort_col = energy_col
    else:
        sort_col = next(
            (c for c in ["target_energy", "binding_energy"]
             if c in docking_results_df.columns),
            None,
        )
    if sort_col is None:
        logger.warning("No energy column found")
        return output_dir

    # 过滤有效对接结果
    valid = docking_results_df[
        docking_results_df.get("docking_success", True) == True
    ].dropna(subset=[sort_col]).copy()

    if valid.empty:
        logger.warning("No valid docking results for report generation")
        return output_dir

    # 1. 理化性质
    df = calculate_cheminformatics_properties(valid)

    # 2. 图表
    generate_score_distribution(df, output_dir, sort_col)
    generate_property_drift(df, output_dir, sort_col)
    generate_chemical_space_tsne(df, output_dir, sort_col)
    scaffold_df = generate_scaffold_enrichment(df, output_dir, sort_col)

    # 3. Top-N 命中分子
    top_hits = df.sort_values(sort_col, ascending=True).head(top_n)

    # 4. Excel
    excel_path = output_dir / f"round_{round_num:03d}_data.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            top_hits.to_excel(writer, sheet_name="Top_Hits", index=False)
            df.to_excel(writer, sheet_name="All_Valid_Results", index=False)
            if scaffold_df is not None:
                scaffold_df.to_excel(writer, sheet_name="Scaffold_Enrichment", index=False)
    except Exception as e:
        logger.warning(f"Excel export failed: {e}")

    # 5. PDF
    try:
        generate_pdf_report(df, top_hits, scaffold_df, output_dir, round_num, summary)
    except Exception as e:
        logger.warning(f"PDF generation failed: {e}")

    # 6. 文本摘要
    _write_text_summary(df, output_dir, round_num, summary, sort_col)

    logger.info(f"Round {round_num} report complete.")
    return output_dir


def _write_text_summary(
    df: pd.DataFrame, output_dir: Path, round_num: int,
    summary: dict, energy_col: str,
):
    """写入文本摘要（自动检测双靶点）"""
    txt_path = output_dir / f"round_{round_num:03d}_summary.txt"
    energies = df[energy_col].dropna()
    has_anti = "anti_energy" in df.columns and df["anti_energy"].notna().any()

    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"  MOLECULAR DIRECTED EVOLUTION — ROUND {round_num} SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Valid Docking Results: {len(df)}\n")

        if has_anti:
            f.write(f"\nTarget Score:\n")
            te = df["target_energy"].dropna()
            f.write(f"  Best (lowest):  {te.min():.2f} kcal/mol\n")
            f.write(f"  Mean:           {te.mean():.2f} kcal/mol\n")
            f.write(f"  Median:         {te.median():.2f} kcal/mol\n")
            f.write(f"  Top 10%:        ≤ {np.percentile(te, 10):.2f} kcal/mol\n")

            f.write(f"\nAnti-Target Score (>0 = no binding):\n")
            ae = df["anti_energy"].dropna()
            f.write(f"  Best (highest): {ae.max():.2f} kcal/mol\n")
            f.write(f"  Mean:           {ae.mean():.2f} kcal/mol\n")
            f.write(f"  % Non-binding:  {(ae >= 0).sum() / len(ae) * 100:.1f}%\n")

            f.write(f"\nSelectivity (Delta = Target - Anti):\n")
            de = df["delta"].dropna()
            f.write(f"  Best (lowest):  {de.min():.2f} kcal/mol\n")
            f.write(f"  Mean:           {de.mean():.2f} kcal/mol\n")
            f.write(f"  Selective (Δ<0): {(de < 0).sum()}/{len(de)} ({(de < 0).sum()/len(de)*100:.1f}%)\n")
            f.write(f"  Top 10%:        ≤ {np.percentile(de, 10):.2f} kcal/mol\n")
        else:
            f.write(f"Best Binding Energy:        {energies.min():.2f} kcal/mol\n")
            f.write(f"Mean Binding Energy:        {energies.mean():.2f} kcal/mol\n")
            f.write(f"Median Binding Energy:      {energies.median():.2f} kcal/mol\n")
            f.write(f"Std Dev:                    {energies.std():.2f} kcal/mol\n")
            f.write(f"  Top 10%:  ≤ {np.percentile(energies, 10):.2f} kcal/mol\n")

        f.write(f"\nSeeds selected for next round: {summary.get('seeds_selected', 'N/A')}\n")

    logger.info(f"  Text summary → {txt_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 自检
    print(f"RDKit available: {_rdkit_available}")
    print(f"reportlab available: {_reportlab_available}")

    # 用已有对接结果测试
    docking_csv = Path(
        "/home/xiayun-huang/Desktop/virtual_screening_pipeline/"
        "output/rounds/round_001/04_docking/docking_results_round1.csv"
    )
    if docking_csv.exists():
        df = pd.read_csv(docking_csv)
        print(f"Loaded {len(df)} results")

        out = Path("output/rounds/round_001/06_report")
        generate_round_report(df, round_num=1, output_dir=out)

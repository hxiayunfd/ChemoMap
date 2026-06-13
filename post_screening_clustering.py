#!/usr/bin/env python3
"""
模块: 筛选后聚类分析（兼容层）
Module: Post-Screening Clustering (Compatibility Wrapper)

此模块已合并到 seed_selector.py 的 PostScreeningClustering 类中。
保留此文件仅为了向后兼容。

请直接使用:
    from seed_selector import PostScreeningClustering
"""

from seed_selector import PostScreeningClustering

# 重新导出以保持兼容
__all__ = ["PostScreeningClustering"]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    print("PostScreeningClustering is now available from seed_selector module.")
    print("Usage: from seed_selector import PostScreeningClustering")

# 🔄 Virtual Screening Pipeline — 项目重组进度

## ✅ 已完成的修复 (2026-06-10)

### 项目结构重组
- [x] 创建 `conda_manager.py` — Conda 环境自动扫描与工具分配
- [x] 重写 `config.py` — 完整配置系统，支持 JSON 覆盖和环境变量
- [x] 创建 `__init__.py` — Python 包结构
- [x] 重构 `main.py` — 完整 CLI，支持 `--pdb-id`, `--interactive`, `--status`
- [x] 重构 `protein_preparator.py` — 整合 CondaManager，添加快捷方法
- [x] 重构 `ligand_preparator.py` — 整合 CondaManager，简化接口
- [x] 重构 `docking_engine.py` — 整合 CondaManager，改进错误处理
- [x] 重构 `property_filter.py` — 整合 CondaManager，添加一站式方法
- [x] 重构 `seed_selector.py` — 合并 PostScreeningClustering 功能，消除代码重复
- [x] 重构 `reinvent_generator.py` — 修复 CLI 调用格式 Bug
- [x] 重构 `pipeline_orchestrator.py` — 修复已知 Bug，惰性模块加载
- [x] 重构 `post_screening_clustering.py` — 改为兼容层
- [x] 重构 `tool_compound_generator.py` — 整合 CondaManager
- [x] 更新 `configs/example_config.json` 和 `configs/reinvent_template.json`

### 已修复的 Bug
- [x] **Bug 1**: Reinvent4 CLI 格式 → 修复为 `-f json <config_file>`
- [x] **Bug 2**: 缺少 `np` (numpy) 导入 → 已在 protein_preparator.py 和 pipeline_orchestrator.py 添加
- [x] **Bug 3**: `analyze_round()` 方法不存在 → 已移除调用，改用 `select_seeds()`
- [x] **Bug 4**: PostScreeningClustering 与 SeedSelector 代码重复 → 已合并到 SeedSelector

## 📦 新项目结构

```
virtual_screening_pipeline/
├── __init__.py                 # 包入口
├── main.py                     # CLI 主入口
├── config.py                   # 全局配置
├── conda_manager.py            # [NEW] Conda 环境管理
├── protein_preparator.py       # PDB 获取/修复/口袋检测/PDBQT
├── ligand_preparator.py        # SMILES → 3D SDF → PDBQT
├── docking_engine.py           # AutoDock-GPU 对接
├── property_filter.py          # 理化性质筛选
├── seed_selector.py            # 种子选择 + 聚类分析 [已合并]
├── reinvent_generator.py       # Reinvent4 分子生成 [Bug 已修复]
├── pipeline_orchestrator.py    # 流水线编排 [Bug 已修复]
├── post_screening_clustering.py # 兼容层 → seed_selector
├── tool_compound_generator.py  # ChEMBL 工具化合物
├── configs/
│   ├── example_config.json     # [已更新]
│   └── reinvent_template.json  # [已更新]
├── script/                     # 原始脚本（保留参考）
├── data/                       # 输入数据
├── logs/                       # 日志
├── output/                     # 输出
└── temp/                       # 临时文件
```

## 🚀 快速使用

```bash
# 查看环境状态
python main.py --status

# 交互式蛋白准备 + 单轮测试
python main.py --pdb-id 4dkl --interactive --test

# 使用本地蛋白文件迭代筛选
python main.py --pdb output/prepared_protein/4dkl.pdbqt \
               --center_x 10 --center_y 25 --center_z 12 \
               --size_x 30 --size_y 30 --size_z 30 \
               --seeds seeds.smi --max-rounds 5

# 使用配置文件
python main.py --config configs/example_config.json
```

## 📋 待办 (可选增强)

- [ ] 添加单元测试
- [ ] 支持 SmiLib 格式配体库输入
- [ ] 添加 AutoGrid4 网格文件 (.gpf) 自动生成
- [ ] 支持多受体并行对接
- [ ] Web UI 监控面板

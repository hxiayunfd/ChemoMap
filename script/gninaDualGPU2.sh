#!/bin/bash

# ================= 1. 动态路径感知区域 =================
BASE_DIR="/home/xiayun-huang/KDproject"

# 严格按照时间从新到旧排序 (-t)，尾部加 / 强制只匹配文件夹，取第一个最新创建的
LATEST_DIR=$(ls -td $BASE_DIR/Enamine_r[0-9]*/ 2>/dev/null | head -n 1)

if [ -z "$LATEST_DIR" ]; then
    echo "❌ 错误: 在 $BASE_DIR 下未找到任何新生成的 Enamine_r* 文件夹！"
    exit 1
fi

# 去除路径末尾可能带有的斜杠，保证后续拼接路径的规范性
LATEST_DIR="${LATEST_DIR%/}"

# 从绝对路径中提取文件夹名（例如 Enamine_r4）
FOLDER_NAME=$(basename "$LATEST_DIR")

# 动态反推对应的输入 CSV、配体目录以及输出文件（全部锁死在工作文件夹内）
CSV_FILE="$LATEST_DIR/${FOLDER_NAME}.csv"
LIGAND_DIR="$LATEST_DIR"
OUTPUT_CSV="$LATEST_DIR/${FOLDER_NAME}_docking_results.csv"

# 受体物理路径保持不变
RECEPTOR_NMDA="/home/xiayun-huang/KDproject/7eu8_1_dock.pdbqt"
RECEPTOR_OPIOD="/home/xiayun-huang/KDproject/4DKL_1_dock.pdbqt"

echo "=================================================="
echo "📁 自动感知到最新工作文件夹: $FOLDER_NAME"
echo "📄 动态锁定文件夹内数据源: ${FOLDER_NAME}.csv"
echo "📝 对接结果 CSV 将导出至: $OUTPUT_CSV"
echo "=================================================="

# ================= 2. Grid Box 参数区域 =================
# 定义 7eu8 (NMDA) 的 Grid Box 参数
NMDA_CENTER_X=131.741; NMDA_CENTER_Y=131.532; NMDA_CENTER_Z=72.669
NMDA_SIZE_X=22.0; NMDA_SIZE_Y=22.0; NMDA_SIZE_Z=22.0

# 定义 4DKL (Opiod) 的 Grid Box 参数
OPIOD_CENTER_X=-28.02; OPIOD_CENTER_Y=-13.33; OPIOD_CENTER_Z=-11.05
OPIOD_SIZE_X=15.0; OPIOD_SIZE_Y=15.0; OPIOD_SIZE_Z=15.0

# ================= 3. 初始化输出 CSV =================
# 注意：原汁原味抓取 Mode 1 的真实 VINA RESULT 打分，表头保持英文字段对齐
echo "compound,smiles,NMDA Affinity,Opiod Affinity,selectivity" > "$OUTPUT_CSV"

# ================= 4. 处理单个分子的函数 =================
process_molecule() {
    local compound_name="$1"
    local smiles="$2"
    local ligand_file="$LIGAND_DIR/${compound_name}.pdbqt"

    if [ ! -f "$ligand_file" ]; then
        echo "警告: 找不到 $compound_name，已跳过。"
        ( flock -x 200; echo "$compound_name,$smiles,,," >> "$OUTPUT_CSV" ) 200>"$OUTPUT_CSV.lock"
        return
    fi

    echo "正在处理: $compound_name ..."

    # 1. NMDA 对接（输出的 Pose 同样规范到工作文件夹下）
    local output_nmda="${LIGAND_DIR}/${compound_name}_7eu8_out.pdbqt"
    gnina -r "$RECEPTOR_NMDA" -l "$ligand_file" -o "$output_nmda" \
        --center_x $NMDA_CENTER_X --center_y $NMDA_CENTER_Y --center_z $NMDA_CENTER_Z \
        --size_x $NMDA_SIZE_X --size_y $NMDA_SIZE_Y --size_z $NMDA_SIZE_Z \
        --device 0 > /dev/null 2>&1
    
    # 纯文本层面精准修复：跳过被初始位置干扰的头部，直接提取表格中 Mode 1 的 VINA RESULT 真实能量
    local nmda_score=$(grep "REMARK VINA RESULT" "$output_nmda" 2>/dev/null | head -n 1 | awk '{print $4}')

    # 2. Opiod 对接（输出的 Pose 同样规范到工作文件夹下）
    local output_opiod="${LIGAND_DIR}/${compound_name}_4DKL_out.pdbqt"
    gnina -r "$RECEPTOR_OPIOD" -l "$ligand_file" -o "$output_opiod" \
        --center_x $OPIOD_CENTER_X --center_y $OPIOD_CENTER_Y --center_z $OPIOD_CENTER_Z \
        --size_x $OPIOD_SIZE_X --size_y $OPIOD_SIZE_Y --size_z $OPIOD_SIZE_Z \
        --device 0 > /dev/null 2>&1
    
    # 纯文本层面精准修复：跳过被初始位置干扰的头部，直接提取表格中 Mode 1 的 VINA RESULT 真实能量
    local opiod_score=$(grep "REMARK VINA RESULT" "$output_opiod" 2>/dev/null | head -n 1 | awk '{print $4}')

    # 3. 计算 selectivity
    local selectivity=""
    if [ -n "$nmda_score" ] && [ -n "$opiod_score" ]; then
        if [ "$(echo "$opiod_score != 0" | bc -l)" -eq 1 ]; then
            selectivity=$(echo "scale=4; $nmda_score / $opiod_score" | bc -l)
        fi
    fi

    # 4. 写入结果（加文件锁）
    ( flock -x 200; echo "$compound_name,$smiles,$nmda_score,$opiod_score,$selectivity" >> "$OUTPUT_CSV" ) 200>"$OUTPUT_CSV.lock"
    
    # 响应你的诉求，删除了原先的 rm -f 逻辑，完整保留 PDBQT Docking Pose 供 PyMOL 调阅
}

# 导出函数和变量给 parallel 使用
export -f process_molecule
export LIGAND_DIR RECEPTOR_NMDA RECEPTOR_OPIOD OUTPUT_CSV
export NMDA_CENTER_X NMDA_CENTER_Y NMDA_CENTER_Z NMDA_SIZE_X NMDA_SIZE_Y NMDA_SIZE_Z
export OPIOD_CENTER_X OPIOD_CENTER_Y OPIOD_CENTER_Z OPIOD_SIZE_X OPIOD_SIZE_Y OPIOD_SIZE_Z

# ================= 5. 任务分发执行 =================
# 核心指令：维持你完全跑得通的原生 parallel 语法结构，确保执行无误
tail -n +2 "$CSV_FILE" | tr -d '\r' | parallel -j 2 --colsep ',' process_molecule {1} {2}

# 清理锁文件
rm -f "$OUTPUT_CSV.lock"

echo "所有化合物对接完成！"

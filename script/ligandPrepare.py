import os
from rdkit import Chem
from rdkit.Chem import AllChem
from concurrent.futures import ProcessPoolExecutor, as_completed

# 根据你的 i9-12900HX 处理器，设置最大并行进程数
MAX_WORKERS = 20

def _prepare_single(args):
    """内部多进程调用的真实执行函数，接收元组参数"""
    name, smiles, output_path = args
    try:
        # 【核心修改】将化合物名称中的空格替换为下划线，确保文件名和内部属性合法
        name = name.replace(" ", "_")
        
        # 1. 从 SMILES 生成 RDKit 分子对象并添加名称
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return name, False

        mol.SetProp("_Name", name)  # 这里的 name 已经是替换过空格的了
        
        # 2. 添加氢原子
        mol = Chem.AddHs(mol)

        # 3. 生成 3D 构象 (使用 ETKDG 算法)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params=params) == -1:
            return name, False

        # 4. 几何优化 (MMFF94 力场)
        AllChem.MMFFOptimizeMolecule(mol)

        # 5. 计算 Gasteiger 电荷
        AllChem.ComputeGasteigerCharges(mol)

        # 6. 写入 SDF 文件 (output_path 里的文件名也会因为 name 的变化而自动变成下划线)
        writer = Chem.SDWriter(str(output_path))
        writer.write(mol)
        writer.close()

        # 简单质检：文件存在且大小正常
        return name, os.path.exists(output_path) and os.path.getsize(output_path) > 200
    except Exception:
        return name, False

def process_to_3d_sdf(name, smiles, output_path):
    """
    对外保留的标准接口，供 proteinQC.py 正常调用
    """
    _, success = _prepare_single((name, smiles, output_path))
    return success

def batch_prepare(compound_list, output_dir):
    """
    批量并行处理配体准备
    """
    print(f"\n[PREPARE] 启动多进程配体准备 (并行数: {MAX_WORKERS})...")
    
    tasks = []
    for name, smiles in compound_list:
        # 提前把名称里的空格换掉，保证路径拼接也是安全的
        safe_name = name.replace(" ", "_")
        output_path = os.path.join(output_dir, f"{safe_name}.sdf")
        tasks.append((safe_name, smiles, output_path))

    success_count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_name = {executor.submit(_prepare_single, task): task[0] for task in tasks}
        
        for i, future in enumerate(as_completed(future_to_name)):
            name, success = future.result()
            if success:
                success_count += 1
                print(f"[PREPARE] ({i+1}/{len(tasks)}) 成功: {name}")
            else:
                print(f"[PREPARE] ({i+1}/{len(tasks)}) 失败: {name}")

    print(f"\n[SUCCESS] 配体准备完成！成功生成 {success_count} 个 SDF 文件，失败 {len(tasks) - success_count} 个。")

if __name__ == "__main__":
    pass

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def rename_file(file_path, pattern, replacement):
    """ 执行重命名逻辑 """
    if pattern in file_path.name:
        new_name = file_path.name.replace(pattern, replacement)
        new_path = file_path.with_name(new_name)
        try:
            file_path.rename(new_path)
        except Exception as e:
            print(f"Error renaming {file_path.name}: {e}")

def process_dataset(root_dir):
    root = Path(root_dir)
    tasks = []

    # 1. 扫描所有文件
    print("正在扫描文件...")
    # 扫描所有子目录下的 rgb 和 poses
    for sub_dir in root.rglob('*'):
        if not sub_dir.is_dir():
            continue
        
        # 处理 RGB 目录
        if sub_dir.name == 'rgb':
            for f in sub_dir.glob('*.color.png'):
                tasks.append((f, '.color', ''))
        
        # 处理 Poses 目录
        elif sub_dir.name == 'poses':
            for f in sub_dir.glob('*.pose.txt'):
                tasks.append((f, '.pose', ''))

    print(f"找到待处理文件: {len(tasks)} 个")

    # 2. 并行执行重命名 (使用多线程提升 I/O 效率)
    print("正在执行并行重命名...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        for task in tasks:
            executor.submit(rename_file, *task)

    print("完成！")

if __name__ == "__main__":
    # 指定你的数据集根目录
    target_dir = "/data/nvme0n1/phw/7scenes"
    process_dataset(target_dir)
import subprocess
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

manifold_plus_bin = "/data/hdd1/storage/junpeng/ws_dditplus/ManifoldPlus/build/manifold"
shapenet_root="/data/hdd1/storage/junpeng/ws_dditplus/DeepSDF/data/ShapeNetCore.v2"

category_name2id = {
    'bottle': '02876657',
    'cup': '',
    'plane':'02691156',
}

if __name__=="__main__":
    output_root = "/data/hdd1/storage/junpeng/ws_dditplus/HyperDiffusion/data/ShapeNet_processed"
    category_path = os.path.join(shapenet_root, category_name2id['plane'])
    category_output_dir = os.path.join(output_root, category_name2id['plane'])

    if not os.path.exists(category_output_dir):
        os.makedirs(category_output_dir)

    def process_file(file_id):
        obj_path = os.path.join(category_path, file_id, "models/model_normalized.obj")
        obj_output_path = os.path.join(category_output_dir, "{}.obj".format(file_id))
        if not os.path.exists(obj_output_path):
            subprocess.run([manifold_plus_bin, "--input", obj_path, "--output", obj_output_path])
        else:
            print("{} existis and will be skipped!".format(obj_output_path))

    if __name__ == '__main__':
        file_ids = os.listdir(category_path)
        num_processes = 10

        # 使用 multiprocessing.Pool 来并行处理文件
        with Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap(process_file, file_ids), total=len(file_ids)))

from glob import glob
import shutil
import os

source_dir = "./90_M1/*"
source_dirs = sorted(glob(source_dir))


gt_dirs = [_ for _ in source_dirs if "gt" in _]

for gt_dir in gt_dirs:
    file_name = os.path.basename(gt_dir).split("_gt")[0]
    related_dirs = [_ for _ in source_dirs if file_name in _ and "gt" not in _ and "$" not in _]

    shortest_name = min(related_dirs, key=lambda x: len(x))
    related_dirs.remove(shortest_name)
    name_len = len(shortest_name)

    if len(related_dirs) == 0:
        print(f"No related directories found for {file_name}. Skipping.")
        continue
    
    min_ = 999999
    target_dir = related_dirs[0]
    for x in related_dirs:
        num = int(x[name_len-3:name_len].split("_")[0])
        if num < min_:
            min_ = num
            target_dir = x

    # print(file_name)
    # print(target_dir)
    # print(related_dirs)
    # print(sorted(related_dirs))

    sem_name = target_dir.split("/")[-1]
    gt_name = gt_dir.split("/")[-1]

    shutil.copy(target_dir, f"./90/sem/{sem_name}")
    shutil.copy(gt_dir, f"./90/label/{gt_name}")




# for i in range(0, len(source_dir)-4, 4):
#     sem = source_dirs[i]
#     gt = source_dirs[i + 2]
#     assert "gt" in gt, f"Expected 'gt' in {gt}"
#     sem_name = sem.split("/")[-1]
#     gt_name = gt.split("/")[-1]
#     print(sem_name, gt_name)
    # shutil.copy(sem, f"./32/sem/{sem_name}")
    # shutil.copy(gt, f"./32/label/{gt_name}")





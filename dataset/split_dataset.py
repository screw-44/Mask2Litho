import shutil
from glob import glob
import os

label_directory = "./32/label/*"
sem_directory = "./32/sem/*"
label_dirs = sorted(glob(label_directory))
sem_dirs = sorted(glob(sem_directory))

paired_dirs = [(i, j) for i, j in zip(label_dirs, sem_dirs)]

train_paired_dirs = paired_dirs[:int(len(paired_dirs) * 0.6)]
val_paired_dirs = paired_dirs[int(len(paired_dirs) * 0.6):]


def save_to_directory(train_paired_dirs, is_train=True):
    if is_train:
        os.makedirs("./32/train/label", exist_ok=True)
        os.makedirs("./32/train/sem", exist_ok=True)
    else:
        os.makedirs("./32/val/label", exist_ok=True)
        os.makedirs("./32/val/sem", exist_ok=True)

    for label_dir, sem_dir in train_paired_dirs:
        label_name = os.path.basename(label_dir)
        sem_name = os.path.basename(sem_dir)
        if is_train:
            shutil.copy(label_dir, f"./32/train/label/{label_name}")
            shutil.copy(sem_dir, f"./32/train/sem/{sem_name}")
        else:
            shutil.copy(label_dir, f"./32/val/label/{label_name}")
            shutil.copy(sem_dir, f"./32/val/sem/{sem_name}")

save_to_directory(train_paired_dirs, is_train=True)
save_to_directory(val_paired_dirs, is_train=False)
print(f"Training data saved to ./32/train/label and ./32/train/sem")
print(f"Validation data saved to ./32/val/label and ./32/val/sem")



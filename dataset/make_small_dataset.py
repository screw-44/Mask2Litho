import os

from glob import glob


dir = "./32_small/train/label/*"

dirs = sorted(glob(dir))

print("dirs:", len(dirs))

for _ in dirs[1000:]:
    os.remove(_)

print("new dirs:", len(glob(dir)))


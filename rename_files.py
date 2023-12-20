import sys
from pathlib import Path
import shutil


if __name__ == '__main__':
    dirname = Path(sys.argv[1])
    start_index = int(sys.argv[2])
    for file in dirname.iterdir():
        print(file)
        shutil.copy(file, file.parent / ("0"*(3 - len(str(start_index))) + str(start_index) + ".mp4"))
        start_index += 1

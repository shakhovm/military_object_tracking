import json
import shutil

import numpy as np
import sys
from pathlib import Path

if __name__ == '__main__':
    file, outsides_dir, matrices_dir = sys.argv[1:]
    outsides_dir = Path(outsides_dir)
    file = Path(file)
    matrices_dir = Path(matrices_dir)
    file_name_s = file.name
    file_name = file_name_s.replace("-s", '')
    matrix_file = file_name.replace("txt", "json")
    with open(str(file.absolute()), 'r') as f:
        data = f.readlines()

    with open(matrices_dir / matrix_file, 'r') as f:
        matrices = json.load(f)
    new_bboxes = []
    for bbox, matrix in zip(data, matrices):
        m = np.zeros((2, 3), np.float32)
        bbox = list(map(float, bbox.split(',')))
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        dx, dy, da = matrix
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy
        A = m[:2, :2]
        A_int = np.linalg.inv(A)
        B = m[:, 2]
        x_y = [np.array([bbox[0], bbox[1]]),
               np.array([bbox[0], bbox[3]]),
               np.array([bbox[2], bbox[1]]),
               np.array([bbox[2], bbox[3]])]
        new_coord = []
        for coord in x_y:
            new_coord.append(A_int @ (coord - B))
        # new_bbox = [min(x_y[0], x_y[])]
        bbox = [min(new_coord[0][0], new_coord[1][0]),
                min(new_coord[0][1], new_coord[1][1]),
                max(new_coord[2][0], new_coord[3][0]),
                max(new_coord[2][1], new_coord[3][1])]

        bbox = [max(0, bbox[0]), max(0, bbox[1]), min(bbox[2], 10000), min(bbox[3], 10000)]
        bbox = [str(bbox[0]), str(bbox[1]), str(bbox[2] - bbox[0]), str(bbox[3] - bbox[1])]
        # [max(x_y[0], 0), max(x_y[1], 0), x_y[0] + bbox[2] - bbox[0], x_y[1] + bbox[3] - bbox[1]]

        new_bboxes.append(",".join(bbox))

    with open(f"{str(file.absolute()).replace('-s', '')}", 'w') as f:
        f.write("\n".join(new_bboxes))
        shutil.copy(str(outsides_dir / file_name_s), str(outsides_dir / file_name))



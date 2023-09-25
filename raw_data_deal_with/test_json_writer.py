import json
import numpy as np


def write_test_json(json_path):
    dict = {}
    test_ex_mat = np.eye(4, 4)
    test_ex_mat[2, 3] = 9  # have an init position to render
    dict["1_1_M"] = test_ex_mat.tolist()
    dict["rotation"] = [1, 0, 0, 0]
    dict["translation"] = [0, 0, 0]
    with open(json_path, 'w') as json_file:
        json.dump(dict, json_file, indent=4)
    return dict


if __name__ == '__main__':
    filename = 'D:/gitwork/NeuS/dynamic_test/test_render.json'
    write_test_json(filename)



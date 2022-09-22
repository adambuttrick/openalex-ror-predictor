import pickle
from os import path


def load_pkl(model_path, pkl):
    with open(path.join(model_path, pkl), "rb") as f:
        pkl = pickle.load(f)
    return pkl

def invert_dict(d):
    return {i: j for j, i in d.items()}
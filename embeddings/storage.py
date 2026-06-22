import numpy as np

def save_semantic(user_id, vector):

    path = f"data/semantic/{user_id}.npy"

    np.save(path, vector)

    return path


def save_steering(user_id, vector):

    path = f"data/steering/{user_id}.npy"

    np.save(path, vector)

    return path


def load_semantic(path):
    return np.load(path)


def load_steering(path):
    return np.load(path)
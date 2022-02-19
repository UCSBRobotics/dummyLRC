import numpy as np

def onlyBinary(data, labels):
    mask = (labels == 0) | (labels == 1)
    return data[mask], labels[mask]
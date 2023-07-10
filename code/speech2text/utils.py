import numpy as np
import pandas


def get_max_len(sentences):
    max_len = 0
    for sentence in sentences:
        if len(sentence) > max_len:
            max_len = len(sentence)

    return max_len

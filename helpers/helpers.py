import tensorflow as tf
import re


def chunk_text(data):
    words = data.lower().replace("\n", "<EOS>").split()
    results = [re.sub(r'[^\w\s]', '', word) for word in words]
    return results


def read_file(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read()


def split_list(s_list):
    half = len(s_list)//2
    return s_list[:half], s_list[half:]

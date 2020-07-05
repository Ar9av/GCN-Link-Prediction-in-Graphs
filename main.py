import tensorflow as tf
import os
import sys
import time
from data_loader import load_data


if __name__ == "__main__": 
    adj, features= load_data("cora")
    print(adj.shape)
    print(features.shape)
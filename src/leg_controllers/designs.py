
import numpy as np

class Params:
    def __init__(self, *params):
        self.s1_L = params[0]        # Extension Spring 1 free length
        self.s1_k = params[1]        # Extension Spring 1 spring rate
        self.s1_Fi = params[2]      # Extension Spring 1 initial tension
        self.s1_r = params[3]        # Extension Spring 1 rest angle
        self.s3_L = params[4]        # Compression Spring free length
        self.s3_k = params[5]        # Compression Spring spring rate 
        self.l1 = params[6]          # Femur length
        self.l2 = params[7]          # Tibia length

def pack(p):
    return np.array([
        p.s1_L, p.s1_k, p.s1_Fi, p.s1_r, p.s3_L, p.s3_k, p.l1, p.l2
        ])

def unpack(x):
    return Params(*x)

    
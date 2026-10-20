# -*- coding: utf-8 -*-
"""Does `--seed` actually control Keras 3 initial weights?

Builds the stock model twice in ONE process and prints a digest of the initial
weights, under three seeding regimes. If the digest changes between the two
builds of a regime, that regime does not control initialisation.
"""
import hashlib
import os
import random
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asla_complete import create_model


def digest(model):
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(np.ascontiguousarray(w, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def build_twice(label, seed_np, seed_tf, seed_py):
    out = []
    for _ in range(2):
        if seed_np:
            np.random.seed(7)
        if seed_tf:
            tf.random.set_seed(7)
        if seed_py:
            random.seed(7)
        out.append(digest(create_model()))
    same = out[0] == out[1]
    print('%-42s %s  %s' % (label, out, 'REPRODUCIBLE' if same else 'DIFFERS'))
    return same


print('regime                                     digests            verdict')
build_twice('np + tf only (the OLD asla_complete)', True, True, False)
build_twice('np + tf + python random (the FIX)', True, True, True)

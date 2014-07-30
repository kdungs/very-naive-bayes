#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.metrics import classification_report

_, _, _, _, truth = np.loadtxt('data/test.dat', unpack=True)
pred = np.loadtxt('predictions.dat')[:-1]

print(classification_report(truth, pred))

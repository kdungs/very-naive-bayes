#!/usr/bin/env python

import numpy as np

CONFIG = {
    'SIG': {
        'label': 0,
        'num': 1000,
        'means': [1.4, 12, 4, 1.3],
        'sigmas': [.3, 2, .5, .3]
    },
    'BKG': {
        'label': 1,
        'num': 1000,
        'means': [1.6, 13, 5, 1.5],
        'sigmas': [.4, 1, .6, .2]
    }
}

for filename in ('train.dat', 'test.dat'):
    with open('data/{}'.format(filename), 'w') as f:
        for name, conf in CONFIG.items():
            events = list(
                map(
                    lambda x: np.random.normal(x[0], x[1], conf['num']),
                    zip(conf['means'], conf['sigmas'])
                )
            )
            events.append([conf['label']] * conf['num'])
            for e in zip(*events):
                f.write('  '.join(map(str, e)) + '\n')

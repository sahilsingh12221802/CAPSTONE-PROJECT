"""Compatibility wrapper.

Primary training implementation now lives in ml/train.py.
"""

import runpy

if __name__ == '__main__':
    runpy.run_module('ml.train', run_name='__main__')

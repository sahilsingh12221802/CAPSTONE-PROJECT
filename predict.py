"""Compatibility wrapper.

Primary inference implementation now lives in ml/predict.py.
"""

import runpy

from ml.predict import classify, load_and_preprocess

if __name__ == '__main__':
    runpy.run_module('ml.predict', run_name='__main__')

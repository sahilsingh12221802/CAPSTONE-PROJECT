"""Compatibility wrapper.

Primary inference implementation now lives in ml/predict.py.
"""

import runpy

from ml.predict import classify as classify  # noqa: F401
from ml.predict import load_and_preprocess as load_and_preprocess  # noqa: F401

__all__ = ['classify', 'load_and_preprocess']

if __name__ == '__main__':
    runpy.run_module('ml.predict', run_name='__main__')

"""Compatibility wrapper.

Primary evaluation implementation now lives in ml/evaluate.py.
"""

import runpy

if __name__ == '__main__':
    runpy.run_module('ml.evaluate', run_name='__main__')

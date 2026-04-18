"""Compatibility wrapper.

Primary backend implementation now lives in backend/app.py.
"""

from backend.app import app

if __name__ == '__main__':
    import uvicorn

    uvicorn.run('backend.app:app', host='127.0.0.1', port=8000, reload=True)

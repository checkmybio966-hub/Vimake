"""Root entrypoint (`uvicorn main:app`) - Vercel ke liye bhi.

Vercel FastAPI ko `api/index.py` ya root ke `app.py / index.py / server.py /
main.py` me dhoondhta hai. Ye file root wale candidate ko cover karti hai;
dono jagah se milta hai to bhi **ek hi `app` object** hai, isliye result same.
"""
from server.app import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

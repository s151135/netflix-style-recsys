"""Run with a clean wheel-installed interpreter, outside the source directory."""
from importlib.resources import files

from fastapi.testclient import TestClient

from recsys import __version__
from recsys.serving.api import app


assert __version__ == "1.0.0"
assert files("recsys.serving").joinpath("demo.html").is_file()
assert files("recsys.serving").joinpath("demo.js").is_file()
with TestClient(app) as client:
    assert client.get("/").status_code == 200
    assert client.get("/assets/demo.js").status_code == 200
    assert client.get("/healthz").status_code == 200
print("Installed wheel and UI assets verified")

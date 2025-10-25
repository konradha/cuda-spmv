import os
import io
import pathlib
import requests
import gzip
import tarfile
import shutil

DEFAULT_MATRICES = [
    ("Williams", "webbase-1M"),
    ("SNAP", "amazon0312"),
    ("SNAP", "roadNet-CA"),
    ("Bova", "rma10"),
    ("GHS_psdef", "bmw7st_1"),
    ("Schmid", "thermal2"),
    ("Andrianov", "net25"),
    ("Rothberg", "cfd1"),
    ("ATandT", "atm"),
    ("Goodwin", "mesh2e1"),
    ("Norris", "torso1"),
    ("DIMACS10", "coPapersDBLP"),
    ("JGD_Kocay", "bddk1"),
    ("DNVS", "shipsec1"),
    ("GHS_indef", "tuma1"),
]


def ensure_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def _is_gzip_magic(b):
    return len(b) >= 2 and b[0] == 0x1F and b[1] == 0x8B


def _valid_mtx(path):
    try:
        with open(path, "rb") as f:
            head = f.read(32)
        return head.startswith(b"%%MatrixMarket")
    except Exception:
        return False


def _http_get(url, timeout=300):
    r = requests.get(url, stream=True, allow_redirects=True, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(str(r.status_code))
    return r


def _write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)


def _save_mtx_gz(buf, out_path):
    with gzip.GzipFile(fileobj=io.BytesIO(buf), mode="rb") as gzf, open(out_path, "wb") as fout:
        shutil.copyfileobj(gzf, fout)


def _extract_mtx_from_tar_gz(buf, out_dir, prefer_name=None):
    with tarfile.open(fileobj=io.BytesIO(buf), mode="r:gz") as tar:
        cand = [m for m in tar.getmembers() if m.isfile()
                and m.name.lower().endswith(".mtx")]
        if not cand:
            raise RuntimeError("no mtx in tar")
        m = None
        if prefer_name:
            for c in cand:
                base = os.path.basename(c.name)
                if base.split(".")[0].lower() == prefer_name.lower():
                    m = c
                    break
        if m is None:
            m = cand[0]
        target = os.path.join(out_dir, os.path.basename(m.name))
        with tar.extractfile(m) as fin, open(target, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return target


def _download_one(group, name, dest_dir):
    base = f"https://sparse.tamu.edu/MM/{group}/{name}"
    mtx_gz = base + ".mtx.gz"
    tar_gz = base + ".tar.gz"
    try:
        r = _http_get(mtx_gz)
        head = r.raw.read(2)
        rest = r.raw.read()
        buf = head + rest
        if not _is_gzip_magic(buf[:2]):
            raise RuntimeError("not gz")
        out_path = os.path.join(dest_dir, f"{group}_{name}.mtx")
        _save_mtx_gz(buf, out_path)
        if not _valid_mtx(out_path):
            os.remove(out_path)
            raise RuntimeError("bad mtx")
        return out_path
    except Exception:
        r2 = _http_get(tar_gz)
        data = r2.content
        path = _extract_mtx_from_tar_gz(data, dest_dir, prefer_name=name)
        if not _valid_mtx(path):
            os.remove(path)
            raise RuntimeError("bad mtx")
        return path


def download(dest_dir, pairs=None, limit=12):
    ensure_dir(dest_dir)
    if pairs is None:
        pairs = DEFAULT_MATRICES[:limit]
    out = []
    for g, n in pairs:
        try:
            out.append(_download_one(g, n, dest_dir))
        except Exception:
            pass
    return out

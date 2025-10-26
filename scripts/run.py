import argparse
import os
import sys
import glob
import csv
import subprocess
import shlex
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def run(cmd, cwd=None, check=True, capture=True):
  r = subprocess.run(cmd if isinstance(cmd, list) else shlex.split(cmd), cwd=cwd, text=True, stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)
  if check and r.returncode != 0:
    msg = r.stdout or ""
    raise SystemExit(f"[cmd failed] {' '.join(cmd if isinstance(cmd, list) else [cmd])}\n{msg}")
  return r.returncode, (r.stdout or "")

def ensure_dir(p):
  os.makedirs(p, exist_ok=True)
  return p

def which(tool):
  for path in os.environ.get("PATH", "").split(os.pathsep):
    cand = os.path.join(path.strip('"'), tool)
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
      return cand
  return None

def discover_datasets(candidates, data_dir, limit=None):
  found = []
  def add(p):
    if os.path.isfile(p):
      found.append(os.path.abspath(p))
  for c in candidates:
    if os.path.isabs(c) and os.path.isfile(c):
      add(c); continue
    if os.path.isfile(c):
      add(c); continue
    p = os.path.join(data_dir, c)
    if os.path.isfile(p):
      add(p); continue
    base = os.path.splitext(os.path.basename(c))[0]
    patterns = [f"**/{base}.mtx", f"**/{base}.npz", f"**/{base}.csr", f"**/{base}.bin", f"**/{base}.*"]
    for pat in patterns:
      for hit in glob.glob(os.path.join(data_dir, pat), recursive=True):
        add(hit)
  if not candidates:
    for pat in ("**/*.mtx", "**/*.npz", "**/*.csr", "**/*.bin"):
      for hit in glob.glob(os.path.join(data_dir, pat), recursive=True):
        add(hit)
  uniq, seen = [], set()
  for p in found:
    if p not in seen:
      uniq.append(p); seen.add(p)
  if limit is not None:
    uniq = uniq[:limit]
  return uniq

def build_all(root):
  exe = os.path.join(root, "bin", "spmv_all")
  if os.path.isfile(exe):
    return exe
  rc, out = run(["make"], cwd=root, check=False)
  if rc != 0 or not os.path.isfile(exe):
    sys.stderr.write(out or "")
    raise SystemExit(f"build failed or missing driver: {exe}")
  return exe

DEFAULT_NCU_METRICS = [
  "dram__bytes_read.sum",
  "dram__bytes_write.sum",
  "dram__throughput.avg.pct_of_peak_sustained_elapsed",
  "l1tex__t_bytes.sum",
  "lts__t_bytes.sum",
  "l1tex__t_sector_hit_rate",
  "lts__t_sector_hit_rate",
  "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
  "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
  "smsp__thread_inst_executed_per_inst_executed.pct",
  "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
  "smsp__warps_issue_stalled_lg_throttle_per_warp_active.pct",
  "smsp__warps_issue_stalled_mio_throttle_per_warp_active.pct",
  "sm__warps_active.avg.pct_of_peak_sustained_active",
]

def run_with_ncu(ncu, exe, matrix_path, warmup, repeat, root, log_dir, kernel_regex=None, extra_metrics=None):
  ensure_dir(log_dir)
  log_csv = os.path.join(log_dir, f"{os.path.basename(matrix_path)}.ncu.csv")
  metrics = list(DEFAULT_NCU_METRICS)
  if extra_metrics:
    for m in extra_metrics.split(","):
      m = m.strip()
      if m:
        metrics.append(m)
  cmd = [ncu, "--csv", "--target-processes", "all", "--profile-from-start", "off", "--set", "full", "--metrics", ",".join(metrics), "--log-file", log_csv]
  if kernel_regex:
    cmd += ["--kernel-name", kernel_regex]
  cmd += ["--", exe, matrix_path, str(warmup), str(repeat)]
  run(cmd, cwd=root, check=True, capture=True)
  with open(log_csv, "r", newline="") as f:
    rows = list(csv.reader(f))
  header_idx = None
  for i, row in enumerate(rows):
    if len(row) >= 3 and row[0] == "ID" and "Metric Name" in row and "Metric Value" in row:
      header_idx = i; break
  metrics_map = {}
  if header_idx is not None:
    for row in rows[header_idx + 1:]:
      if len(row) >= 3:
        name = row[1].strip(); val = row[2].strip()
        if name and val:
          metrics_map[name] = val
  metrics_map["__matrix__"] = os.path.basename(matrix_path)
  return metrics_map, log_csv

SASS_PATTERNS = {
  "FFMA": r"\bFFMA\b",
  "IMAD": r"\bIMAD\b",
  "LDG": r"\bLDG(\.[A-Z0-9_]+)?\b",
  "SHFL": r"\bSHFL(\.[A-Z0-9_]+)?\b",
}
PTX_PATTERNS = {
  "ld.global": r"\bld\.global(\.[a-z0-9\.]+)?\b",
  "ld.shared": r"\bld\.shared(\.[a-z0-9\.]+)?\b",
  "bra": r"\bbra\b",
  "setp": r"\bsetp(\.[a-z0-9]+)?\b",
  "reg_decl": r"\.reg\s+\.[a-z0-9]+\s+%\w+<(\d+)>",
}

def cuobjdump_available():
  return which("cuobjdump") is not None

def dump_ptx_sass(exe_path, out_dir):
  ensure_dir(out_dir)
  ptx_txt = os.path.join(out_dir, "ptx.txt")
  sass_txt = os.path.join(out_dir, "sass.txt")
  _, ptx = run(["cuobjdump", "--dump-ptx", exe_path], check=True)
  with open(ptx_txt, "w") as f:
    f.write(ptx)
  _, sass = run(["cuobjdump", "--dump-sass", exe_path], check=True)
  with open(sass_txt, "w") as f:
    f.write(sass)
  return ptx, sass, ptx_txt, sass_txt

def count_patterns(text, patterns):
  return {k: len(re.findall(rgx, text)) for k, rgx in patterns.items()}

def extract_ptx_register_pressure(ptx_text):
  regs = re.findall(PTX_PATTERNS["reg_decl"], ptx_text)
  vals = [int(x) for x in regs] if regs else []
  return max(vals) if vals else None

def run_driver_all(exe, matrix, warmup, repeat, root):
  r = subprocess.run([exe, matrix, str(warmup), str(repeat)], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  if r.returncode != 0:
    raise SystemExit(r.stdout)
  return r.stdout

def per_matrix_bar(df, value_col, out_path, ylabel):
  piv = df.pivot_table(index="matrix", columns="method", values=value_col)
  ax = piv.plot(kind="bar", figsize=(12, 6))
  ax.set_ylabel(ylabel)
  ax.set_xlabel("matrix")
  ax.get_figure().tight_layout()
  ax.get_figure().savefig(out_path, dpi=300)

def speedup_vs_best_cusparse(df, value_col, out_path, ylabel):
  piv = df.pivot_table(index="matrix", columns="method", values=value_col)
  cus_cols = [c for c in piv.columns if c.startswith("cusparse_")]
  if not cus_cols:
    return
  best_cusp = piv[cus_cols].max(axis=1)
  sp = piv.divide(best_cusp, axis=0)
  ax = sp.plot(kind="bar", figsize=(12, 6))
  ax.set_ylabel(ylabel)
  ax.set_xlabel("matrix")
  ax.axhline(1.0)
  ax.get_figure().tight_layout()
  ax.get_figure().savefig(out_path, dpi=300)

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--download-defaults", action="store_true")
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--warmup", type=int, default=10)
  ap.add_argument("--repeat", type=int, default=50)
  ap.add_argument("--paths", nargs="*", default=[])
  ap.add_argument("--out", default=None)
  ap.add_argument("--data-root", default=None)
  ap.add_argument("--ncu", action="store_true")
  ap.add_argument("--ncu-kernel", default=None)
  ap.add_argument("--ncu-metrics", default=None)
  ap.add_argument("--dump-ptx-sass", dest="dump_ptx_sass", action="store_true")
  args = ap.parse_args()

  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  data_dir = args.data_root or os.path.join(root, "data")
  out_dir = ensure_dir(os.path.join(root, "out"))
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  run_out_dir = ensure_dir(os.path.join(out_dir, f"allkernels_{stamp}"))

  if args.download_defaults:
    from dataset import download as ds_download
    ensure_dir(data_dir)
    ds_download(data_dir, limit=args.limit if args.limit else 12)

  paths = discover_datasets(args.paths, data_dir, limit=args.limit)
  if not paths:
    raise SystemExit(f"no datasets resolved under {data_dir} (inputs: {args.paths})")

  exe = build_all(root)

  ncu = which("ncu") if args.ncu else None
  if args.ncu and not ncu:
    raise SystemExit("Nsight Compute CLI (ncu) not found in PATH.")

  combined_rows = []
  header = None
  for pth in paths:
    out = run_driver_all(exe, pth, args.warmup, args.repeat, root).strip()
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
      continue
    if header is None:
      header = lines[0]
    rows = [ln for ln in lines[1:] if "," in ln]
    combined_rows.extend(rows)

  if header is None or not combined_rows:
    raise SystemExit("no results from benchmark driver")

  out_csv = args.out or os.path.join(run_out_dir, "results_all.csv")
  with open(out_csv, "w") as f:
    f.write(header + "\n")
    for r in combined_rows:
      f.write(r + "\n")

  try:
    df = pd.read_csv(out_csv)
    df["gbytes_per_s"] = df["gbps"]
    per_matrix_bar(df, "gflops", os.path.join(run_out_dir, "gflops_all.png"), "GFLOP/s")
    per_matrix_bar(df, "gbytes_per_s", os.path.join(run_out_dir, "gbps_all.png"), "GB/s")
    speedup_vs_best_cusparse(df, "gflops", os.path.join(run_out_dir, "speedup_vs_bestcusparse_all.png"), "Speedup vs best cuSPARSE (GFLOP/s)")
  except Exception as e:
    sys.stderr.write(f"[plot warn] {e}\n")

  ncu_rows = []
  ncu_logs_dir = None
  if ncu:
    ncu_logs_dir = ensure_dir(os.path.join(run_out_dir, "ncu_logs"))
    for pth in paths:
      met, _ = run_with_ncu(ncu=ncu, exe=exe, matrix_path=pth, warmup=args.warmup, repeat=args.repeat, root=root, log_dir=ncu_logs_dir, kernel_regex=args.ncu_kernel, extra_metrics=args.ncu_metrics)
      ncu_rows.append(met)
    if ncu_rows:
      def to_num(x):
        if isinstance(x, str) and x.endswith("%"):
          x = x[:-1]
        try:
          return float(x)
        except Exception:
          return x
      all_keys = sorted(set().union(*[set(r.keys()) for r in ncu_rows]))
      records = []
      for r in ncu_rows:
        rec = {k: r.get(k, "") for k in all_keys}
        for k, v in rec.items():
          rec[k] = to_num(v)
        records.append(rec)
      ncu_df = pd.DataFrame.from_records(records)
      ncu_df.to_csv(os.path.join(run_out_dir, "ncu_metrics_all.csv"), index=False)

  sassptx_csv = None
  if args.dump_ptx_sass:
    if cuobjdump_available():
      dump_dir = ensure_dir(os.path.join(run_out_dir, "objdump"))
      ptx_text, sass_text, _, _ = dump_ptx_sass(exe, dump_dir)
      sass_counts = count_patterns(sass_text, SASS_PATTERNS)
      ptx_counts = count_patterns(ptx_text, PTX_PATTERNS)
      reg_pressure = extract_ptx_register_pressure(ptx_text)
      row = {"binary": os.path.basename(exe)}
      row.update({f"sass_{k}": v for k, v in sass_counts.items()})
      row.update({f"ptx_{k}": v for k, v in ptx_counts.items()})
      row["ptx_reg_pressure_max"] = reg_pressure if reg_pressure is not None else ""
      sassptx_csv = os.path.join(run_out_dir, "sass_ptx_all.csv")
      pd.DataFrame([row]).to_csv(sassptx_csv, index=False)

  print("== artifacts ==")
  print(out_csv)
  if ncu_rows:
    print(os.path.join(run_out_dir, "ncu_metrics_all.csv"))
    if ncu_logs_dir:
      print(ncu_logs_dir)
  if sassptx_csv:
    print(sassptx_csv)
  print(run_out_dir)

if __name__ == "__main__":
  main()


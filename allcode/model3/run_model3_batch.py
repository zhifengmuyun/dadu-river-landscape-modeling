#!/usr/bin/env python3
"""
Batch runner for the 3 x 3 model3 Badlands experiments.

For each uplift height this script:
1. regenerates fault displacement CSV files with dlsf matching xshf/anhf geometry,
2. generates newtec1.csv/newtec2.csv/newtec3.csv/tecleft.csv/tecdown.csv
   using the same arithmetic as model3.m,
3. updates model3.xml for erodibility and output folder name,
4. runs Dadu_run.py in the current Linux/VSCode environment.
"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


UPLIFT_HEIGHTS = [(1, 300.0), (2, 400.0), (3, 500.0)]
ERODIBILITIES = [(1, "1e-6"), (2, "2e-6"), (3, "3e-6")]

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent


def replace_dlsf_field(source: str, field: str, value: str) -> str:
    pattern = re.compile(
        rf"('name'\s*:\s*['\"]dlsf['\"].*?'{re.escape(field)}'\s*:\s*)[-+0-9.eE]+",
        re.DOTALL,
    )
    updated, count = pattern.subn(lambda m: f"{m.group(1)}{value}", source, count=1)
    if count != 1:
        raise RuntimeError(f"Could not find dlsf {field} in fault_displacement_v4.py")
    return updated


def replace_xml_tag(source: str, tag: str, value: str) -> str:
    pattern = re.compile(rf"(<{tag}>)(.*?)(</{tag}>)", re.DOTALL)
    updated, count = pattern.subn(lambda m: f"{m.group(1)}{value}{m.group(3)}", source, count=1)
    if count != 1:
        raise RuntimeError(f"Could not find <{tag}> in model3.xml")
    return updated


def write_text_file(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def run_checked(command: list[str], cwd: Path, log_file: Path | None = None, dry_run: bool = False) -> None:
    print(f"\n$ {' '.join(shlex.quote(str(part)) for part in command)}")
    print(f"  cwd: {cwd}")
    if dry_run:
        return

    if log_file is None:
        subprocess.run(command, cwd=cwd, check=True)
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8", newline="") as log:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()

    if rc != 0:
        raise subprocess.CalledProcessError(rc, command)


def fault_csv_only_source(source: str) -> str:
    marker = "\n# ============================================================\n# 4."
    cut = source.find(marker)
    if cut == -1:
        raise RuntimeError("Could not find the plotting section in fault_displacement_v4.py")
    save_block = r'''

# ============================================================
# 4. 保存CSV；model3批处理不保存速度场图
# ============================================================
def save_badlands_lcol(filename, arr):
    np.savetxt(filename, arr.ravel(order='C'), fmt='%.6e')

save_badlands_lcol('total_u.csv', U_total)
save_badlands_lcol('total_v.csv', V_total)
save_badlands_lcol('total_w.csv', W_total)

save_badlands_lcol('xshf_u.csv', fault_fields['xshf']['u'])
save_badlands_lcol('xshf_v.csv', fault_fields['xshf']['v'])

save_badlands_lcol('anhf_u.csv', fault_fields['anhf']['u'])
save_badlands_lcol('anhf_v.csv', fault_fields['anhf']['v'])

save_badlands_lcol('dlsf_u.csv', fault_fields['dlsf']['u'])
save_badlands_lcol('dlsf_v.csv', fault_fields['dlsf']['v'])
save_badlands_lcol('dlsf_w.csv', fault_fields['dlsf']['w'])

print("Saved fault displacement CSV files for model3")
'''
    return source[:cut] + save_block


def run_fault_script(model_dir: Path, dry_run: bool = False) -> None:
    source_path = model_dir / "fault_displacement_v4.py"
    original = source_path.read_text(encoding="utf-8")
    batch_path = model_dir / "_fault_displacement_v4_model3_batch.py"
    batch_source = replace_dlsf_field(original, "rake", "0.0")
    batch_source = replace_dlsf_field(batch_source, "dip", "90.0")
    batch_source = replace_dlsf_field(batch_source, "dip_direction", "1")
    batch_source = fault_csv_only_source(batch_source)

    if dry_run:
        print(f"Would write temporary fault script: {batch_path}")
        print("Would set dlsf rake=0.0, dip=90.0, dip_direction=1")
        return

    write_text_file(batch_path, batch_source)
    try:
        run_checked([sys.executable, str(batch_path.name)], cwd=model_dir)
    finally:
        try:
            batch_path.unlink()
        except FileNotFoundError:
            pass


def read_column(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    return np.asarray(data, dtype=float).reshape(-1)


def save_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, fmt="%.10g", delimiter=" ", newline="\r\n")


def save_column(path: Path, data: np.ndarray) -> None:
    np.savetxt(path, data.reshape(-1), fmt="%.10g", newline="\r\n")


def generate_model3_tectonics(model_dir: Path, uplift_height: float) -> None:
    nx = 200
    ny = 1000
    n = nx * ny
    t = 1e6

    dispx_xshf = read_column(model_dir / "xshf_u.csv") / 1e3 * t
    dispy_xshf = read_column(model_dir / "xshf_v.csv") / 1e3 * t
    dispx_anhf = read_column(model_dir / "anhf_u.csv") / 1e3 * t
    dispy_anhf = read_column(model_dir / "anhf_v.csv") / 1e3 * t
    dispx_dlsf = read_column(model_dir / "dlsf_u.csv") / 1e3 * t
    dispy_dlsf = read_column(model_dir / "dlsf_v.csv") / 1e3 * t

    dispz1 = 250.0 * np.ones(n)
    dispz2 = np.zeros(n)
    dispz3 = np.zeros(n)

    # Same 1-based indexing as the MATLAB model3.m loops.
    i = np.arange(1, n + 1)
    row_floor = np.floor(i / nx)
    col_mod = np.mod(i, nx)

    uplift_mask = (row_floor < 500) & ((col_mod > 100) | (col_mod == 0))
    drop_mask = (row_floor > 470) & (row_floor < 480) & ((col_mod > 160) | (col_mod == 0))
    dispz2[uplift_mask] = uplift_height
    dispz3[drop_mask] = -2500.0

    participe = np.zeros(n)
    first = i < n / 3
    second = (~first) & (i < n / 2)
    participe[first] = 0.9
    participe[second] = 0.9 - 0.3 * (row_floor[second] - 333) / (n / 400 - 333)
    participe[~(first | second)] = 0.6

    participe1 = np.where((row_floor < 500) & (col_mod < 100), 2.0, 1.0)

    base_x = dispx_xshf + dispx_anhf
    base_y = dispy_xshf + dispy_anhf
    fault_x = base_x + dispx_dlsf
    fault_y = base_y + dispy_dlsf

    disp1 = np.column_stack((base_x, base_y, dispz1))
    disp2 = np.column_stack((base_x, base_y, dispz1 + dispz2))
    disp3 = 1.5 * np.column_stack((fault_x, fault_y, dispz1 + dispz2))
    disp4 = (2999.0 / 2000.0) * np.column_stack((fault_x, fault_y, dispz1))
    disp5 = (1.0 / 2000.0) * np.column_stack((fault_x, fault_y, dispz1 + 2000.0 * dispz3))

    save_matrix(model_dir / "newtec1.csv", disp1)
    save_matrix(model_dir / "newtec2.csv", disp2)
    save_matrix(model_dir / "newtec3.csv", disp3)
    save_matrix(model_dir / "tecleft.csv", disp4)
    save_matrix(model_dir / "tecdown.csv", disp5)
    save_column(model_dir / "participe.csv", participe)
    save_column(model_dir / "participe1.csv", participe1)


def badlands_command(args: argparse.Namespace) -> list[str]:
    if args.badlands_command:
        return shlex.split(args.badlands_command)
    return [sys.executable, "Dadu_run.py"]


def prepare_output_dir(path: Path, overwrite: bool, skip_existing: bool) -> bool:
    if not path.exists():
        return True
    if skip_existing:
        print(f"Skip existing output: {path}")
        return False
    if overwrite:
        shutil.rmtree(path)
        return True
    raise FileExistsError(
        f"Output folder already exists: {path}. Use --overwrite or --skip-existing."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 9 model3 uplift/erodibility Badlands experiments."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--badlands-command",
        help="Custom command used to run Badlands. Default: current Python runs Dadu_run.py.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete existing model3_x-y outputs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip model3_x-y outputs that exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    xml_path = model_dir / "model3.xml"
    original_xml = xml_path.read_text(encoding="utf-8")
    command = badlands_command(args)

    if args.overwrite and args.skip_existing:
        raise RuntimeError("Use only one of --overwrite and --skip-existing.")

    print(f"Model dir:    {model_dir}")

    try:
        print("\n=== Fault displacement group: dlsf matches xshf/anhf ===")
        run_fault_script(model_dir, dry_run=args.dry_run)

        for uplift_index, uplift_height in UPLIFT_HEIGHTS:
            print(f"\n=== Uplift group {uplift_index}: dispz2 height = {uplift_height:g} ===")
            if not args.dry_run:
                generate_model3_tectonics(model_dir, uplift_height)

            for erod_index, erodibility in ERODIBILITIES:
                out_name = f"model3_{uplift_index}-{erod_index}"
                output_dir = model_dir / out_name
                if not prepare_output_dir(output_dir, args.overwrite, args.skip_existing):
                    continue

                xml_text = replace_xml_tag(original_xml, "erodibility", erodibility)
                xml_text = replace_xml_tag(xml_text, "outfolder", out_name)
                print(
                    f"\n--- Run {out_name}: dispz2 height={uplift_height:g}, "
                    f"erodibility={erodibility} ---"
                )

                if not args.dry_run:
                    write_text_file(xml_path, xml_text)

                log_file = model_dir / "batch_logs" / f"{out_name}.log"
                run_checked(command, cwd=model_dir, log_file=log_file, dry_run=args.dry_run)

                if not args.dry_run and output_dir.exists():
                    shutil.copy2(xml_path, output_dir / "model3.xml")
                    for name in (
                        "newtec1.csv",
                        "newtec2.csv",
                        "newtec3.csv",
                        "tecleft.csv",
                        "tecdown.csv",
                        "participe.csv",
                        "participe1.csv",
                    ):
                        shutil.copy2(model_dir / name, output_dir / name)

    finally:
        if not args.dry_run:
            write_text_file(xml_path, original_xml)

    print("\nAll requested model3 runs are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

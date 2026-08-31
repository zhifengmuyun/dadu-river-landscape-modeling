#!/usr/bin/env python3
"""
Batch runner for the 3 x 3 model1 Badlands experiments.

For each rake value this script:
1. runs fault_displacement_v4.py with the dlsf rake changed,
2. generates newtec1.csv/newtec2.csv/newtec3.csv using the same arithmetic as model1.m,
3. copies the tectonic CSV files into this model1 folder,
4. updates Daduhejif.xml for erodibility and output folder name,
5. runs Dadu_run.py in the current Linux/VSCode environment.
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


RAKES = [(1, 16.0), (2, 33.0), (3, 55.0)]
ERODIBILITIES = [(1, "1e-6"), (2, "2e-6"), (3, "3e-6")]

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_SHARED_ROOT = Path("/mnt/c/user/Wenyan/landscapeinput")


def replace_dlsf_rake(source: str, rake: float) -> str:
    pattern = re.compile(
        r"('name'\s*:\s*['\"]dlsf['\"].*?'rake'\s*:\s*)[-+0-9.eE]+",
        re.DOTALL,
    )
    updated, count = pattern.subn(lambda m: f"{m.group(1)}{rake:.1f}", source, count=1)
    if count != 1:
        raise RuntimeError("Could not find the dlsf rake entry in fault_displacement_v4.py")
    return updated


def replace_xml_tag(source: str, tag: str, value: str) -> str:
    pattern = re.compile(rf"(<{tag}>)(.*?)(</{tag}>)", re.DOTALL)
    updated, count = pattern.subn(lambda m: f"{m.group(1)}{value}{m.group(3)}", source, count=1)
    if count != 1:
        raise RuntimeError(f"Could not find <{tag}> in Daduhejif.xml")
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


def run_fault_displacement(settings_dir: Path, rake: float, dry_run: bool = False) -> None:
    source_path = settings_dir / "fault_displacement_v4.py"
    original = source_path.read_text(encoding="utf-8")
    batch_path = settings_dir / "_fault_displacement_v4_batch.py"
    batch_source = replace_dlsf_rake(original, rake)

    if dry_run:
        print(f"Would write temporary fault script: {batch_path}")
        print(f"Would set dlsf rake to {rake:.1f}")
        return

    write_text_file(batch_path, batch_source)
    try:
        run_checked([sys.executable, str(batch_path.name)], cwd=settings_dir)
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


def generate_model1_tectonics(settings_dir: Path) -> None:
    nx = 200
    ny = 1000
    n = nx * ny
    t = 1e6

    dispx_xshf = read_column(settings_dir / "xshf_u.csv") / 1e3 * t
    dispy_xshf = read_column(settings_dir / "xshf_v.csv") / 1e3 * t
    dispx_anhf = read_column(settings_dir / "anhf_u.csv") / 1e3 * t
    dispy_anhf = read_column(settings_dir / "anhf_v.csv") / 1e3 * t
    dispx_dlsf = read_column(settings_dir / "dlsf_u.csv") / 1e3 * t
    dispy_dlsf = read_column(settings_dir / "dlsf_v.csv") / 1e3 * t
    dispz_dlsf = read_column(settings_dir / "dlsf_w.csv") / 1e3 * t

    dispz1 = 250.0 * np.ones(n)

    # Badlands order: lower-left to upper-right. Rainfall varies by y row.
    i = np.arange(1, n + 1)
    row_floor = np.floor(i / nx)
    participe = np.zeros(n)
    first = i < n / 3
    second = (~first) & (i < n / 2)
    participe[first] = 0.9
    participe[second] = 0.9 - 0.3 * (row_floor[second] - 333) / (n / 400 - 333)
    participe[~(first | second)] = 0.6

    participe1 = np.where((row_floor < 500) & (np.mod(i, nx) < 100), 2.0, 1.0)

    disp1 = np.column_stack((dispx_xshf + dispx_anhf, dispy_xshf + dispy_anhf, dispz1))
    disp2 = np.column_stack((dispx_xshf + dispx_anhf, dispy_xshf + dispy_anhf, dispz1))
    disp3 = np.column_stack(
        (
            dispx_xshf + dispx_anhf + dispx_dlsf,
            dispy_xshf + dispy_anhf + dispy_dlsf,
            dispz1 + dispz_dlsf,
        )
    )

    save_matrix(settings_dir / "newtec1.csv", disp1)
    save_matrix(settings_dir / "newtec2.csv", disp2)
    save_matrix(settings_dir / "newtec3.csv", disp3)
    save_column(settings_dir / "participe.csv", participe)
    save_column(settings_dir / "participe1.csv", participe1)


def copy_tectonic_files(settings_dir: Path, model_dir: Path) -> None:
    for name in (
        "newtec1.csv",
        "newtec2.csv",
        "newtec3.csv",
        "participe.csv",
        "participe1.csv",
    ):
        src = settings_dir / name
        dst = model_dir / name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)


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
        description="Run the 9 model1 rake/erodibility Badlands experiments."
    )
    parser.add_argument(
        "--settings-dir",
        type=Path,
        help="Folder containing fault_displacement_v4.py. Default: model-dir.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument(
        "--badlands-command",
        help=(
            "Custom command used to run Badlands. Default: current Python runs Dadu_run.py."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete existing model1_x-y outputs.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip model1_x-y outputs that exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    settings_dir = args.settings_dir.resolve() if args.settings_dir else model_dir
    xml_path = model_dir / "Daduhejif.xml"
    original_xml = xml_path.read_text(encoding="utf-8")

    args.shared_root = args.shared_root.resolve()
    command = badlands_command(args)

    if args.overwrite and args.skip_existing:
        raise RuntimeError("Use only one of --overwrite and --skip-existing.")

    print(f"Settings dir: {settings_dir}")
    print(f"Model dir:    {model_dir}")
    print(f"Shared root:  {args.shared_root}")

    try:
        for rake_index, rake in RAKES:
            print(f"\n=== Rake group {rake_index}: dlsf rake = {rake:g} ===")
            run_fault_displacement(settings_dir, rake, dry_run=args.dry_run)
            if not args.dry_run:
                generate_model1_tectonics(settings_dir)
                copy_tectonic_files(settings_dir, model_dir)

            for erod_index, erodibility in ERODIBILITIES:
                out_name = f"model1_{rake_index}-{erod_index}"
                output_dir = model_dir / out_name
                if not prepare_output_dir(output_dir, args.overwrite, args.skip_existing):
                    continue

                xml_text = replace_xml_tag(original_xml, "erodibility", erodibility)
                xml_text = replace_xml_tag(xml_text, "outfolder", out_name)
                print(
                    f"\n--- Run {out_name}: rake={rake:g}, erodibility={erodibility} ---"
                )

                if not args.dry_run:
                    write_text_file(xml_path, xml_text)

                log_file = model_dir / "batch_logs" / f"{out_name}.log"
                run_checked(command, cwd=model_dir, log_file=log_file, dry_run=args.dry_run)

                if not args.dry_run and output_dir.exists():
                    shutil.copy2(xml_path, output_dir / "Daduhejif.xml")
                    for name in (
                        "newtec1.csv",
                        "newtec2.csv",
                        "newtec3.csv",
                        "participe.csv",
                        "participe1.csv",
                    ):
                        shutil.copy2(model_dir / name, output_dir / name)

    finally:
        if not args.dry_run:
            write_text_file(xml_path, original_xml)

    print("\nAll requested model1 runs are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

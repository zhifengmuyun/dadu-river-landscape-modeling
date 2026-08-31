#!/usr/bin/env python3
"""
Batch runner for model1 rake=55 plus Gongga/Tuowu uplift and model3-style dam/capture.

This script:
1. runs model1/fault_displacement_v4.py with dlsf rake changed to 55,
2. runs model2/uplift_with_noise.py with the requested peak rates,
3. generates newtec1/newtec2/newtec3/newtec4 for the integrated timeline,
4. copies the tectonic CSV files into this model1 folder,
5. updates Daduhejif.xml for tectonic timing, erodibility, and output folder name,
6. builds a localized precipitation enhancement east of Gongga near the river bend,
7. runs Dadu_run.py in the current Linux/VSCode environment.

Integrated tectonic timeline:
- 0-1 Ma   : background uplift only                                -> newtec1
- 1-2 Ma   : background + Gongga uplift                            -> newtec2
- 2-3.5 Ma : background + DLSF reverse + Gongga + Tuowu + dam     -> newtec3
- 3.5-5 Ma : background + DLSF reverse + Gongga + Tuowu + capture  -> newtec4

The model3 contribution is simplified to:
- a right-boundary dam uplift active only during 2-3.5 Ma,
- a boundary-fed Minjiang subsidence corridor active during 3.5-5 Ma.

No short-lived tecdown/tecleft breach is used in this version.
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


RAKE = (3, 40.0)
PEAK_RATE_H = (2, 0.25)
PEAK_RATE_V = (1, 0.5)
ERODIBILITIES = [(1, "1e-6")]

DEFAULT_GONGGA_EXTRA_PEAK_RATE = 0.05
DEFAULT_GONGGA_EXTRA_PEAK_X = 80.0
DEFAULT_GONGGA_EXTRA_PEAK_Y = 450.0
DEFAULT_GONGGA_EXTRA_PEAK_SIGMA_X = 8.0
DEFAULT_GONGGA_EXTRA_PEAK_SIGMA_Y = 12.0
DEFAULT_GONGGA_RAIN_FACTOR = 0.7
DEFAULT_GONGGA_RAIN_RANGE_SIGMA = 3.0

# Local precipitation enhancement east of Gongga, centred on the major river bend.
# Coordinates are grid coordinates; for the current 1-km grid they are also km.
DEFAULT_BEND_RAIN_X = 115.0
DEFAULT_BEND_RAIN_Y = 500.0
DEFAULT_BEND_RAIN_FACTOR = 1.5
DEFAULT_BEND_RAIN_SIGMA_X = 40.0
DEFAULT_BEND_RAIN_SIGMA_Y = 50.0
DEFAULT_BEND_EAST_TRANSITION = 10.0

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_UPLIFT_DIR = DEFAULT_MODEL_DIR.parent / "model2"

DEFAULT_BACKGROUND_UPLIFT = 250.0
MODEL3_STAGE_DURATION_MYR = 1.5

DEFAULT_RIGHT_UPLIFT_HEIGHT = 1500.0
DEFAULT_RIGHT_UPLIFT_X_MIN = 70.0
DEFAULT_RIGHT_UPLIFT_X_MAX = 199.0
DEFAULT_RIGHT_UPLIFT_CENTER_Y = 500.0
DEFAULT_RIGHT_UPLIFT_SIGMA_Y = 13.0

DEFAULT_MINJIANG_SUBSIDENCE_HEIGHT = -1500.0
DEFAULT_MINJIANG_X_MIN = 70.0
DEFAULT_MINJIANG_X_MAX = 199.0
DEFAULT_MINJIANG_CENTER_Y = 500.0
DEFAULT_MINJIANG_SIGMA_Y = 5.0


def replace_dlsf_rake(source: str, rake: float) -> str:
    pattern = re.compile(
        r"('name'\s*:\s*['\"]dlsf['\"].*?'rake'\s*:\s*)[-+0-9.eE]+",
        re.DOTALL,
    )
    updated, count = pattern.subn(lambda m: f"{m.group(1)}{rake:.1f}", source, count=1)
    if count != 1:
        raise RuntimeError("Could not find the dlsf rake entry in fault_displacement_v4.py")
    return updated


def replace_assignment(source: str, name: str, value: float) -> str:
    pattern = re.compile(rf"^({re.escape(name)}\s*=\s*)[-+0-9.eE]+", re.MULTILINE)
    updated, count = pattern.subn(lambda m: f"{m.group(1)}{value:g}", source, count=1)
    if count != 1:
        raise RuntimeError(f"Could not find assignment for {name} in uplift_with_noise.py")
    return updated


def replace_xml_tag(source: str, tag: str, value: str) -> str:
    pattern = re.compile(rf"(<{tag}>)(.*?)(</{tag}>)", re.DOTALL)
    updated, count = pattern.subn(lambda m: f"{m.group(1)}{value}{m.group(3)}", source, count=1)
    if count != 1:
        raise RuntimeError(f"Could not find <{tag}> in Daduhejif.xml")
    return updated


def replace_tectonic_block(source: str, block: str) -> str:
    pattern = re.compile(r"(<tectonic>)(.*?)(</tectonic>)", re.DOTALL)
    updated, count = pattern.subn(lambda m: f"{m.group(1)}\n{block}\n{m.group(3)}", source, count=1)
    if count != 1:
        raise RuntimeError("Could not find <tectonic> block in Daduhejif.xml")
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
    batch_path = settings_dir / "_fault_displacement_v4_rake55_plus_uplift_model3_dam_capture_batch.py"
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


def run_uplift_script(
    uplift_dir: Path,
    peak_h: float,
    peak_v: float,
    gongga_extra_peak_rate: float,
    gongga_extra_peak_x: float,
    gongga_extra_peak_y: float,
    gongga_extra_peak_sigma_x: float,
    gongga_extra_peak_sigma_y: float,
    dry_run: bool = False,
) -> None:
    source_path = uplift_dir / "uplift_with_noise.py"
    original = source_path.read_text(encoding="utf-8")
    batch_path = uplift_dir / "_uplift_with_noise_rake55_plus_uplift_model3_dam_capture_batch.py"
    batch_source = replace_assignment(original, "peak_rate_v", peak_v)
    batch_source = replace_assignment(batch_source, "peak_rate_h", peak_h)
    batch_source = replace_assignment(batch_source, "extra_peak_rate", gongga_extra_peak_rate)
    batch_source = replace_assignment(batch_source, "extra_peak_x", gongga_extra_peak_x)
    batch_source = replace_assignment(batch_source, "extra_peak_y", gongga_extra_peak_y)
    batch_source = replace_assignment(batch_source, "extra_peak_sigma_x", gongga_extra_peak_sigma_x)
    batch_source = replace_assignment(batch_source, "extra_peak_sigma_y", gongga_extra_peak_sigma_y)
    plot_section = batch_source.find("\n# ===========================\n# 7.")
    if plot_section != -1:
        batch_source = (
            batch_source[:plot_section]
            + "\nprint('Saved uplift_v_arm_200k.csv and uplift_h_arm_200k.csv')\n"
        )

    if dry_run:
        print(f"Would write temporary uplift script: {batch_path}")
        print(f"Would set peak_rate_h={peak_h:g}, peak_rate_v={peak_v:g}")
        print(
            "Would set Gongga extra peak: "
            f"rate={gongga_extra_peak_rate:g} mm/a, "
            f"center=({gongga_extra_peak_x:g}, {gongga_extra_peak_y:g}) km, "
            f"sigma=({gongga_extra_peak_sigma_x:g}, {gongga_extra_peak_sigma_y:g}) km"
        )
        return

    write_text_file(batch_path, batch_source)
    try:
        run_checked([sys.executable, str(batch_path.name)], cwd=uplift_dir)
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


def token(value: float | str) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def x_uniform_y_gaussian_field(
    n: int,
    nx: int,
    height: float,
    x_min: float,
    x_max: float,
    center_y: float,
    sigma_y: float,
    name: str,
    y_half: str | None = None,
) -> np.ndarray:
    if x_min > x_max:
        raise ValueError(f"{name} X bounds are invalid.")
    if sigma_y <= 0:
        raise ValueError(f"{name} sigma_y must be positive.")
    if y_half not in (None, "positive", "negative"):
        raise ValueError(f"{name} y_half must be None, 'positive', or 'negative'.")

    idx = np.arange(n)
    x = idx % nx
    y = idx // nx
    field = np.zeros(n)
    mask = (x >= x_min) & (x <= x_max)
    if y_half == "positive":
        mask &= y >= center_y
    elif y_half == "negative":
        mask &= y <= center_y
    y_factor = np.exp(-0.5 * ((y - center_y) / sigma_y) ** 2)
    field[mask] = height * y_factor[mask]
    return field


def right_x_uniform_y_gaussian_uplift(
    n: int,
    nx: int,
    height: float,
    x_min: float,
    x_max: float,
    center_y: float,
    sigma_y: float,
) -> np.ndarray:
    return x_uniform_y_gaussian_field(
        n, nx, height, x_min, x_max, center_y, sigma_y, "Right uplift"
    )


def minjiang_x_uniform_y_gaussian_subsidence(
    n: int,
    nx: int,
    height: float,
    x_min: float,
    x_max: float,
    center_y: float,
    sigma_y: float,
) -> np.ndarray:
    return x_uniform_y_gaussian_field(
        n,
        nx,
        height,
        x_min,
        x_max,
        center_y,
        sigma_y,
        "Minjiang subsidence",
        y_half="positive",
    )


def model12_tectonic_block() -> str:
    return """   <!--  Is 3D displacements on ? (1:on - 0:off). Default is 0. -->
    <disp3d>1</disp3d>
    <merge3d>100.</merge3d>
    <time3d>10000.</time3d>
    <events>3</events>
    <disp>
      <dstart>0.</dstart>
      <dend>1000000</dend>
      <dfile>newtec1.csv</dfile>
    </disp>
    <disp>
      <dstart>1000000.</dstart>
      <dend>2000000</dend>
      <dfile>newtec2.csv</dfile>
    </disp>
    <disp>
      <dstart>2000000.</dstart>
      <dend>5000000</dend>
      <dfile>newtec3.csv</dfile>
    </disp>"""


def generate_combined_tectonics(
    settings_dir: Path,
    uplift_dir: Path,
    background_uplift: float,
    use_model3: bool,
    right_uplift_height: float,
    right_uplift_x_min: float,
    right_uplift_x_max: float,
    right_uplift_center_y: float,
    right_uplift_sigma_y: float,
    minjiang_subsidence_height: float,
    minjiang_x_min: float,
    minjiang_x_max: float,
    minjiang_center_y: float,
    minjiang_sigma_y: float,
    gongga_rain_factor: float,
    gongga_rain_range_sigma: float,
    bend_rain_x: float,
    bend_rain_y: float,
    bend_rain_factor: float,
    bend_rain_sigma_x: float,
    bend_rain_sigma_y: float,
    bend_east_transition: float,
) -> None:
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
    gongga_uplift = read_column(uplift_dir / "uplift_v_arm_200k.csv") / 1e3 * t
    tuowu_uplift = read_column(uplift_dir / "uplift_h_arm_200k.csv") / 1e3 * t

    dispz1 = background_uplift * np.ones(n)

    if use_model3:
        right_uplift = right_x_uniform_y_gaussian_uplift(
            n,
            nx,
            right_uplift_height,
            right_uplift_x_min,
            right_uplift_x_max,
            right_uplift_center_y,
            right_uplift_sigma_y,
        )
        minjiang_subsidence = minjiang_x_uniform_y_gaussian_subsidence(
            n,
            nx,
            minjiang_subsidence_height,
            minjiang_x_min,
            minjiang_x_max,
            minjiang_center_y,
            minjiang_sigma_y,
        )
    else:
        right_uplift = np.zeros(n)
        minjiang_subsidence = np.zeros(n)

    i = np.arange(1, n + 1)
    pa_scalar = 1.0
    row_floor = np.floor(i / nx)
    participe = np.zeros(n)
    first = i < n / 3
    second = (~first) & (i < n / 2)
    participe[first] = 0.9 * pa_scalar
    participe[second] = (0.9 - 0.3 * (row_floor[second] - 333) / (n / 400 - 333)) * pa_scalar
    participe[~(first | second)] = 0.6 * pa_scalar

    # Base topographic precipitation before the Gongga rain-shadow reduction.
    participe_base = participe.copy()

    idx = np.arange(n)
    x = idx % nx
    y = idx // nx
    gongga_origin_x = 70.0
    gongga_origin_y = 400.0
    gongga_top_y = ny - 1.0
    gongga_sigma_y400 = 25.0
    gongga_sigma_y500 = 15.0
    gongga_sigma_field = np.where(
        y <= 400.0,
        gongga_sigma_y400,
        np.where(
            y >= 500.0,
            gongga_sigma_y500,
            gongga_sigma_y400
            - (y - 400.0) * (gongga_sigma_y400 - gongga_sigma_y500) / 100.0,
        ),
    )
    closest_y_on_gongga = np.clip(y, gongga_origin_y, gongga_top_y)
    dx_from_gongga = np.maximum(x - gongga_origin_x, 0.0)
    dy_from_gongga = y - closest_y_on_gongga
    gongga_vertical_dist = np.sqrt(dx_from_gongga**2 + dy_from_gongga**2)
    gongga_rain_reduction = (1.0 - gongga_rain_factor) * np.exp(
        -0.5 * (gongga_vertical_dist / gongga_sigma_field) ** 2
    )
    if gongga_rain_range_sigma > 0:
        gongga_rain_reduction = np.where(
            gongga_vertical_dist <= gongga_rain_range_sigma * gongga_sigma_field,
            gongga_rain_reduction,
            0.0,
        )

    # Original topographic precipitation after the Gongga rain-shadow reduction.
    participe_reduced = participe_base * (1.0 - gongga_rain_reduction)

    # ------------------------------------------------------------
    # Local broad precipitation enhancement east of Gongga.
    # The target at the Gaussian centre is:
    #     precipitation = bend_rain_factor * participe_base
    # Outside the enhancement zone, the original reduced field is retained.
    # ------------------------------------------------------------
    if bend_rain_factor < 0:
        raise ValueError("bend_rain_factor must be non-negative.")
    if bend_rain_sigma_x <= 0 or bend_rain_sigma_y <= 0:
        raise ValueError("bend-rain sigma values must be positive.")
    if bend_east_transition <= 0:
        raise ValueError("bend_east_transition must be positive.")

    bend_gaussian = np.exp(
        -0.5
        * (
            ((x - bend_rain_x) / bend_rain_sigma_x) ** 2
            + ((y - bend_rain_y) / bend_rain_sigma_y) ** 2
        )
    )

    # Smoothly suppress the enhancement west of the Gongga reference line.
    east_gate = np.clip(
        (x - gongga_origin_x) / bend_east_transition,
        0.0,
        1.0,
    )
    bend_rain_weight = bend_gaussian * east_gate

    participe_target = participe_base * bend_rain_factor
    participe = participe_reduced + bend_rain_weight * (
        participe_target - participe_reduced
    )

    print(
        "Bend precipitation enhancement: "
        f"center=({bend_rain_x:g}, {bend_rain_y:g}) km, "
        f"factor={bend_rain_factor:g}, "
        f"sigma=({bend_rain_sigma_x:g}, {bend_rain_sigma_y:g}) km, "
        f"east transition={bend_east_transition:g} km, "
        f"final range={participe.min():.4f}-{participe.max():.4f} m/yr"
    )

    participe1 = np.where((row_floor < 500) & (np.mod(i, nx) < 100), 2.0, 1.0)

    base_x = dispx_xshf + dispx_anhf
    base_y = dispy_xshf + dispy_anhf
    fault_x = base_x + dispx_dlsf
    fault_y = base_y + dispy_dlsf
    fault_z = dispz1 + dispz_dlsf + gongga_uplift + tuowu_uplift

    if use_model3:
        stage3_factor = MODEL3_STAGE_DURATION_MYR
        stage4_factor = MODEL3_STAGE_DURATION_MYR
    else:
        # Preserve the original model1+2 batch behavior when model3 is disabled.
        stage3_factor = 1.0
        stage4_factor = 1.0

    disp1 = np.column_stack((base_x, base_y, dispz1))
    disp2 = np.column_stack((base_x, base_y, dispz1 + gongga_uplift))
    disp3 = np.column_stack(
        (
            stage3_factor * fault_x,
            stage3_factor * fault_y,
            stage3_factor * fault_z + stage3_factor * right_uplift,
        )
    )
    if use_model3:
        disp4 = np.column_stack(
            (
                stage4_factor * fault_x,
                stage4_factor * fault_y,
                stage4_factor * fault_z + stage4_factor * minjiang_subsidence,
            )
        )
    else:
        disp4 = disp3.copy()

    save_matrix(settings_dir / "newtec1.csv", disp1)
    save_matrix(settings_dir / "newtec2.csv", disp2)
    save_matrix(settings_dir / "newtec3.csv", disp3)
    save_matrix(settings_dir / "newtec4.csv", disp4)
    save_column(settings_dir / "participe_base.csv", participe_base)
    save_column(settings_dir / "participe_gongga_reduced.csv", participe_reduced)
    save_column(settings_dir / "bend_rain_weight.csv", bend_rain_weight)
    save_column(settings_dir / "participe.csv", participe)
    save_column(settings_dir / "participe1.csv", participe1)
    save_column(settings_dir / "right_dam_uplift.csv", stage3_factor * right_uplift)
    save_column(settings_dir / "minjiang_subsidence.csv", stage4_factor * minjiang_subsidence)


def copy_inputs_to_model(settings_dir: Path, uplift_dir: Path, model_dir: Path) -> None:
    for name in (
        "newtec1.csv",
        "newtec2.csv",
        "newtec3.csv",
        "newtec4.csv",
        "participe.csv",
        "participe_base.csv",
        "participe_gongga_reduced.csv",
        "bend_rain_weight.csv",
        "participe1.csv",
        "right_dam_uplift.csv",
        "minjiang_subsidence.csv",
    ):
        src = settings_dir / name
        dst = model_dir / name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    for name in ("uplift_v_arm_200k.csv", "uplift_h_arm_200k.csv"):
        src = uplift_dir / name
        dst = model_dir / name

        # uplift-dir 与 model-dir 相同时，文件已经在目标位置。
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)


def tectonic_block() -> str:
    return """   <!--  Is 3D displacements on ? (1:on - 0:off). Default is 0. -->
    <disp3d>1</disp3d>
    <merge3d>100.</merge3d>
    <time3d>10000.</time3d>
    <events>4</events>
    <disp>
      <dstart>0.</dstart>
      <dend>1000000</dend>
      <dfile>newtec1.csv</dfile>
    </disp>
    <disp>
      <dstart>1000000.</dstart>
      <dend>2000000</dend>
      <dfile>newtec2.csv</dfile>
    </disp>
    <disp>
      <dstart>2000000.</dstart>
      <dend>3500000</dend>
      <dfile>newtec3.csv</dfile>
    </disp>
    <disp>
      <dstart>3500000.</dstart>
      <dend>5000000</dend>
      <dfile>newtec4.csv</dfile>
    </disp>"""


def tectonic_block_for(use_model3: bool) -> str:
    return tectonic_block() if use_model3 else model12_tectonic_block()


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
        description="Run model1 dlsf rake=55 plus Gongga/Tuowu uplift and model3 dam/capture."
    )
    parser.add_argument(
        "--settings-dir",
        type=Path,
        help="Folder containing model1 fault_displacement_v4.py. Default: model-dir.",
    )
    parser.add_argument("--uplift-dir", type=Path, default=DEFAULT_UPLIFT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--peak-rate-h",
        type=float,
        default=PEAK_RATE_H[1],
        help="Tuowu uplift peak_rate_h. Default: 0.05.",
    )
    parser.add_argument(
        "--peak-rate-v",
        type=float,
        default=PEAK_RATE_V[1],
        help="Gongga uplift peak_rate_v. Default: 0.4.",
    )
    parser.add_argument(
        "--gongga-extra-peak-rate",
        type=float,
        default=DEFAULT_GONGGA_EXTRA_PEAK_RATE,
        help="Peak uplift rate of the extra 2D Gaussian Gongga peak in mm/a. Default: 0.05.",
    )
    parser.add_argument(
        "--gongga-extra-peak-x",
        type=float,
        default=DEFAULT_GONGGA_EXTRA_PEAK_X,
        help="X center of the extra 2D Gaussian Gongga peak in km. Default: 80.",
    )
    parser.add_argument(
        "--gongga-extra-peak-y",
        type=float,
        default=DEFAULT_GONGGA_EXTRA_PEAK_Y,
        help="Y center of the extra 2D Gaussian Gongga peak in km. Default: 450.",
    )
    parser.add_argument(
        "--gongga-extra-peak-sigma-x",
        type=float,
        default=DEFAULT_GONGGA_EXTRA_PEAK_SIGMA_X,
        help="X-direction sigma of the extra 2D Gaussian Gongga peak in km. Default: 8.",
    )
    parser.add_argument(
        "--gongga-extra-peak-sigma-y",
        type=float,
        default=DEFAULT_GONGGA_EXTRA_PEAK_SIGMA_Y,
        help="Y-direction sigma of the extra 2D Gaussian Gongga peak in km. Default: 12.",
    )
    parser.add_argument(
        "--gongga-rain-factor",
        type=float,
        default=DEFAULT_GONGGA_RAIN_FACTOR,
        help="Multiplier applied to precipitation inside the Gongga vertical uplift arm. Default: 0.7.",
    )
    parser.add_argument(
        "--gongga-rain-range-sigma",
        type=float,
        default=DEFAULT_GONGGA_RAIN_RANGE_SIGMA,
        help="Truncate the Gongga precipitation Gaussian tail beyond this many sigma. Use <=0 for no truncation. Default: 3.",
    )
    parser.add_argument(
        "--bend-rain-x",
        type=float,
        default=DEFAULT_BEND_RAIN_X,
        help="X coordinate of the local precipitation enhancement centre in km. Default: 115.",
    )
    parser.add_argument(
        "--bend-rain-y",
        type=float,
        default=DEFAULT_BEND_RAIN_Y,
        help="Y coordinate of the local precipitation enhancement centre in km. Default: 500.",
    )
    parser.add_argument(
        "--bend-rain-factor",
        type=float,
        default=DEFAULT_BEND_RAIN_FACTOR,
        help="Target precipitation multiplier relative to the base topographic precipitation at the enhancement centre. Default: 1.5.",
    )
    parser.add_argument(
        "--bend-rain-sigma-x",
        type=float,
        default=DEFAULT_BEND_RAIN_SIGMA_X,
        help="X-direction sigma of the local precipitation enhancement in km. Default: 40.",
    )
    parser.add_argument(
        "--bend-rain-sigma-y",
        type=float,
        default=DEFAULT_BEND_RAIN_SIGMA_Y,
        help="Y-direction sigma of the local precipitation enhancement in km. Default: 50.",
    )
    parser.add_argument(
        "--bend-east-transition",
        type=float,
        default=DEFAULT_BEND_EAST_TRANSITION,
        help="Width of the smooth east-only gate measured from x=70 km. Default: 10 km.",
    )
    parser.add_argument(
        "--background-uplift",
        type=float,
        default=DEFAULT_BACKGROUND_UPLIFT,
        help="Uniform background uplift in m/Myr. Default: 250.",
    )
    parser.add_argument(
        "--right-uplift-height",
        type=float,
        default=DEFAULT_RIGHT_UPLIFT_HEIGHT,
        help="Peak value of the right dam uplift in m/Myr. Default: 1500.",
    )
    parser.add_argument(
        "--right-uplift-x-min",
        type=float,
        default=DEFAULT_RIGHT_UPLIFT_X_MIN,
        help="Minimum X coordinate of the dam uplift X-uniform range in km. Default: 70.",
    )
    parser.add_argument(
        "--right-uplift-x-max",
        type=float,
        default=DEFAULT_RIGHT_UPLIFT_X_MAX,
        help="Maximum X coordinate of the dam uplift X-uniform range in km. Default: 199.",
    )
    parser.add_argument(
        "--right-uplift-center-y",
        type=float,
        default=DEFAULT_RIGHT_UPLIFT_CENTER_Y,
        help="Y center of the dam uplift Gaussian decay in km. Default: 500.",
    )
    parser.add_argument(
        "--right-uplift-sigma-y",
        type=float,
        default=DEFAULT_RIGHT_UPLIFT_SIGMA_Y,
        help="Y-direction sigma of the dam uplift Gaussian decay in km. Default: 13.",
    )
    parser.add_argument(
        "--minjiang-subsidence-height",
        type=float,
        default=DEFAULT_MINJIANG_SUBSIDENCE_HEIGHT,
        help="Peak value of the Minjiang subsidence in m/Myr. Use negative values. Default: -1500.",
    )
    parser.add_argument(
        "--minjiang-x-min",
        type=float,
        default=DEFAULT_MINJIANG_X_MIN,
        help="Minimum X coordinate of the Minjiang subsidence X-uniform range in km. Default: 70.",
    )
    parser.add_argument(
        "--minjiang-x-max",
        type=float,
        default=DEFAULT_MINJIANG_X_MAX,
        help="Maximum X coordinate of the Minjiang subsidence X-uniform range in km. Default: 199.",
    )
    parser.add_argument(
        "--minjiang-center-y",
        type=float,
        default=DEFAULT_MINJIANG_CENTER_Y,
        help="Y center of the Minjiang subsidence Gaussian decay in km. Default: 500.",
    )
    parser.add_argument(
        "--minjiang-sigma-y",
        type=float,
        default=DEFAULT_MINJIANG_SIGMA_Y,
        help="Y-direction sigma of the Minjiang subsidence Gaussian decay in km. Default: 5.",
    )
    parser.add_argument(
        "--badlands-command",
        help="Custom command used to run Badlands. Default: current Python runs Dadu_run.py.",
    )
    model3_group = parser.add_mutually_exclusive_group()
    model3_group.add_argument(
        "--model3",
        dest="use_model3",
        action="store_true",
        default=True,
        help="Enable the added model3 dam uplift and Minjiang capture. Default.",
    )
    model3_group.add_argument(
        "--no-model3",
        dest="use_model3",
        action="store_false",
        help="Disable all added model3 terms and restore the original model1+2 three-stage timing.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate tectonic CSV files and update nothing else.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete existing combined output.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip combined output if it exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    settings_dir = args.settings_dir.resolve() if args.settings_dir else model_dir
    uplift_dir = args.uplift_dir.resolve()
    xml_path = model_dir / "Daduhejif.xml"
    original_xml = xml_path.read_text(encoding="utf-8")
    command = badlands_command(args)

    if args.overwrite and args.skip_existing:
        raise RuntimeError("Use only one of --overwrite and --skip-existing.")

    rake_index, rake = RAKE
    h_index = PEAK_RATE_H[0]
    v_index = PEAK_RATE_V[0]

    print(f"Settings dir: {settings_dir}")
    print(f"Uplift dir:   {uplift_dir}")
    print(f"Model dir:    {model_dir}")
    print(f"Use model3:   {args.use_model3}")
    print(
        "Gongga precipitation reduction: "
        f"factor={args.gongga_rain_factor:g}, "
        f"Gaussian tail cutoff={args.gongga_rain_range_sigma:g} sigma"
    )
    print(
        "River-bend precipitation enhancement: "
        f"center=({args.bend_rain_x:g}, {args.bend_rain_y:g}) km, "
        f"factor={args.bend_rain_factor:g}, "
        f"sigma=({args.bend_rain_sigma_x:g}, {args.bend_rain_sigma_y:g}) km, "
        f"east transition={args.bend_east_transition:g} km"
    )
    if args.use_model3:
        print(
            "Right dam uplift, uniform in X and Gaussian in Y: "
            f"height={args.right_uplift_height:g} m/Myr, "
            f"x=[{args.right_uplift_x_min:g}, {args.right_uplift_x_max:g}] km, "
            f"center_y={args.right_uplift_center_y:g} km, "
            f"sigma_y={args.right_uplift_sigma_y:g} km"
        )
        print(
            "Minjiang subsidence, uniform in X and Gaussian in Y: "
            f"height={args.minjiang_subsidence_height:g} m/Myr, "
            f"x=[{args.minjiang_x_min:g}, {args.minjiang_x_max:g}] km, "
            f"center_y={args.minjiang_center_y:g} km, "
            f"sigma_y={args.minjiang_sigma_y:g} km"
        )
    else:
        print("Model3 dam uplift and Minjiang subsidence are disabled.")

    try:
        print(f"\n=== Fault displacement: model1 dlsf rake = {rake:g} ===")
        run_fault_displacement(settings_dir, rake, dry_run=args.dry_run)

        print(
            f"\n=== Uplift: peak_rate_h={args.peak_rate_h:g}, "
            f"peak_rate_v={args.peak_rate_v:g} ==="
        )
        print(
            "Extra Gongga 2D peak: "
            f"rate={args.gongga_extra_peak_rate:g} mm/a, "
            f"center=({args.gongga_extra_peak_x:g}, {args.gongga_extra_peak_y:g}) km, "
            f"sigma=({args.gongga_extra_peak_sigma_x:g}, {args.gongga_extra_peak_sigma_y:g}) km"
        )
        run_uplift_script(
            uplift_dir,
            args.peak_rate_h,
            args.peak_rate_v,
            args.gongga_extra_peak_rate,
            args.gongga_extra_peak_x,
            args.gongga_extra_peak_y,
            args.gongga_extra_peak_sigma_x,
            args.gongga_extra_peak_sigma_y,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            generate_combined_tectonics(
                settings_dir,
                uplift_dir,
                args.background_uplift,
                args.use_model3,
                args.right_uplift_height,
                args.right_uplift_x_min,
                args.right_uplift_x_max,
                args.right_uplift_center_y,
                args.right_uplift_sigma_y,
                args.minjiang_subsidence_height,
                args.minjiang_x_min,
                args.minjiang_x_max,
                args.minjiang_center_y,
                args.minjiang_sigma_y,
                args.gongga_rain_factor,
                args.gongga_rain_range_sigma,
                args.bend_rain_x,
                args.bend_rain_y,
                args.bend_rain_factor,
                args.bend_rain_sigma_x,
                args.bend_rain_sigma_y,
                args.bend_east_transition,
            )
            copy_inputs_to_model(settings_dir, uplift_dir, model_dir)

        if args.prepare_only:
            print("\nPrepared tectonic input files in model dir:")
            print(f"  {model_dir / 'newtec1.csv'}")
            print(f"  {model_dir / 'newtec2.csv'}")
            print(f"  {model_dir / 'newtec3.csv'}")
            print(f"  {model_dir / 'newtec4.csv'}")
            print(f"  {model_dir / 'participe.csv'}")
            print(f"  {model_dir / 'participe_base.csv'}")
            print(f"  {model_dir / 'participe_gongga_reduced.csv'}")
            print(f"  {model_dir / 'bend_rain_weight.csv'}")
            print(f"  {model_dir / 'right_dam_uplift.csv'}")
            print(f"  {model_dir / 'minjiang_subsidence.csv'}")
            print("\nBadlands was not run.")
            return 0

        for erod_index, erodibility in ERODIBILITIES:
            model3_label = "m3on" if args.use_model3 else "m3off"
            out_name = (
                f"model1_r{rake_index}_g{token(args.peak_rate_v)}_t{token(args.peak_rate_h)}_"
                f"{model3_label}_dam{token(args.right_uplift_height)}_mjs{token(args.minjiang_subsidence_height)}_"
                f"brf{token(args.bend_rain_factor)}_brx{token(args.bend_rain_x)}_bry{token(args.bend_rain_y)}_"
                f"bsx{token(args.bend_rain_sigma_x)}_bsy{token(args.bend_rain_sigma_y)}_e{erod_index}"
            )
            output_dir = model_dir / out_name
            if not prepare_output_dir(output_dir, args.overwrite, args.skip_existing):
                continue

            xml_text = replace_tectonic_block(original_xml, tectonic_block_for(args.use_model3))
            xml_text = replace_xml_tag(xml_text, "erodibility", erodibility)
            xml_text = replace_xml_tag(xml_text, "outfolder", out_name)
            print(
                f"\n--- Run {out_name}: rake={rake:g}, "
                f"peak_rate_h={args.peak_rate_h:g}, peak_rate_v={args.peak_rate_v:g}, "
                f"use_model3={args.use_model3}, "
                f"right_uplift={args.right_uplift_height:g}, "
                f"minjiang_subsidence={args.minjiang_subsidence_height:g}, "
                f"bend_rain_factor={args.bend_rain_factor:g}, "
                f"bend_rain_center=({args.bend_rain_x:g}, {args.bend_rain_y:g}), "
                f"bend_rain_sigma=({args.bend_rain_sigma_x:g}, {args.bend_rain_sigma_y:g}), "
                f"erodibility={erodibility} ---"
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
                    "newtec4.csv",
                    "participe.csv",
                    "participe_base.csv",
                    "participe_gongga_reduced.csv",
                    "bend_rain_weight.csv",
                    "participe1.csv",
                    "uplift_v_arm_200k.csv",
                    "uplift_h_arm_200k.csv",
                    "right_dam_uplift.csv",
                    "minjiang_subsidence.csv",
                ):
                    shutil.copy2(model_dir / name, output_dir / name)

    finally:
        if not args.dry_run:
            write_text_file(xml_path, original_xml)

    print("\nCombined model1 + model2 + model3 dam/capture run is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

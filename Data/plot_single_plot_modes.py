import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

REQ = ["frame_number", "range_bin", "doppler_bin", "signal_strength"]


def find_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob("range_doppler.csv"))


def activity_name(csv_path: Path) -> str:
    parts = csv_path.parent.parts
    if len(parts) >= 2:
        return parts[-2]
    return csv_path.parent.name


def read_longform(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if all(c in df.columns for c in REQ):
        out = df[REQ].copy()
    else:
        raw = pd.read_csv(csv_path, header=None)
        if raw.shape[1] >= 5:
            raw = raw.iloc[:, :5].copy()
            raw.columns = ["timestamp", *REQ]
        elif raw.shape[1] == 4:
            raw.columns = REQ
        else:
            raise ValueError(f"Unsupported format: {csv_path}")
        out = raw[REQ].copy()

    out[REQ] = out[REQ].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=REQ)
    out["frame_number"] = out["frame_number"].astype(np.int64)
    out["range_bin"] = out["range_bin"].astype(np.int64)
    out["doppler_bin"] = out["doppler_bin"].astype(np.int64)
    out["signal_strength"] = out["signal_strength"].astype(np.float32)
    return out


def frame_maps(df: pd.DataFrame, shape: tuple[int, int]) -> tuple[list[np.ndarray], list[int]]:
    maps = []
    frame_nums = []
    r_bins, d_bins = shape

    for fno, g in df.groupby("frame_number", sort=True):
        m = np.zeros((r_bins, d_bins), dtype=np.float32)
        r = g["range_bin"].to_numpy(dtype=np.int32)
        d = g["doppler_bin"].to_numpy(dtype=np.int32)
        s = g["signal_strength"].to_numpy(dtype=np.float32)
        m[r, d] = s
        maps.append(m)
        frame_nums.append(int(fno))
    return maps, frame_nums


def collect_maps(csv_paths: list[Path]) -> tuple[list[np.ndarray], list[int], tuple[int, int]]:
    all_df = []
    max_r = 0
    max_d = 0

    for p in csv_paths:
        df = read_longform(p)
        if df.empty:
            continue
        all_df.append(df)
        max_r = max(max_r, int(df["range_bin"].max()))
        max_d = max(max_d, int(df["doppler_bin"].max()))

    if not all_df:
        raise ValueError("No valid frames found")

    shape = (max_r + 1, max_d + 1)
    all_maps = []
    all_frame_nums = []

    for df in all_df:
        maps, fnums = frame_maps(df, shape)
        all_maps.extend(maps)
        all_frame_nums.extend(fnums)

    return all_maps, all_frame_nums, shape


def remove_noisy_maps(maps: list[np.ndarray], frame_nums: list[int]) -> tuple[list[np.ndarray], list[int], list[int]]:
    if len(maps) < 5:
        return maps, frame_nums, []

    means = np.array([m.mean() for m in maps], dtype=np.float32)
    stds = np.array([m.std() for m in maps], dtype=np.float32)
    peaks = np.array([m.max() for m in maps], dtype=np.float32)

    med_mean, med_std, med_peak = np.median(means), np.median(stds), np.median(peaks)
    mad_mean = np.median(np.abs(means - med_mean))
    mad_std = np.median(np.abs(stds - med_std))
    mad_peak = np.median(np.abs(peaks - med_peak))

    def rz(v: np.ndarray, med: float, mad: float) -> np.ndarray:
        if mad < 1e-9:
            return np.zeros_like(v)
        return np.abs((v - med) / (1.4826 * mad))

    z_mean = rz(means, med_mean, mad_mean)
    z_std = rz(stds, med_std, mad_std)
    z_peak = rz(peaks, med_peak, mad_peak)

    mask = (z_peak > 12.0) | (peaks > 10000) | (stds > 2000) | ((z_mean > 12.0) & (z_std > 12.0))

    kept_maps = [m for i, m in enumerate(maps) if not mask[i]]
    kept_nums = [f for i, f in enumerate(frame_nums) if not mask[i]]
    removed_nums = [f for i, f in enumerate(frame_nums) if mask[i]]
    return kept_maps, kept_nums, removed_nums


def plot_sum_rd(
    maps: list[np.ndarray],
    title: str,
    max_range: float,
    max_velocity: float,
    save_path: Path | None,
    no_show: bool,
) -> None:
    summed = np.sum(np.stack(maps), axis=0)
    shifted = np.fft.fftshift(summed, axes=1)

    bg = np.median(shifted, axis=0, keepdims=True)
    shifted = np.clip(shifted - bg, 0, None)
    shifted = np.log1p(shifted)

    vmin = float(np.percentile(shifted, 5))
    vmax = float(np.percentile(shifted, 99))

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        shifted.T,
        origin="lower",
        aspect="auto",
        cmap="jet",
        extent=[0.0, max_range, -max_velocity, max_velocity],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Doppler [m/s]")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Summed Signal Strength")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")

    if no_show:
        plt.close(fig)
    else:
        plt.show()


def plot_all_frames_single(
    maps: list[np.ndarray],
    title: str,
    max_range: float,
    save_path: Path | None,
    no_show: bool,
) -> None:
    # Single plot that contains whole frame sequence: range-time map.
    # For each frame, keep strongest doppler response per range bin.
    rt = np.stack([np.max(np.fft.fftshift(m, axes=1), axis=1) for m in maps], axis=0)
    rt = np.log1p(rt)

    vmin = float(np.percentile(rt, 5))
    vmax = float(np.percentile(rt, 99))

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        rt.T,
        origin="lower",
        aspect="auto",
        cmap="jet",
        extent=[0, len(maps), 0.0, max_range],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Range [m]")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Max Doppler Response (log scale)")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")

    if no_show:
        plt.close(fig)
    else:
        plt.show()


def _playback_process_frame(
    frame: np.ndarray,
    grid_size: int,
    min_value: float,
    max_value: float,
) -> np.ndarray:
    shifted = np.fft.fftshift(frame, axes=1)
    desired_rows = max(grid_size, shifted.shape[0])
    desired_cols = max(grid_size, shifted.shape[1])

    scale_y = desired_rows / shifted.shape[0]
    scale_x = desired_cols / shifted.shape[1]
    interp = zoom(shifted, (scale_y, scale_x), order=1)

    denom = max(max_value - min_value, 1e-9)
    normalized = (interp - min_value) / denom
    return normalized.astype(np.float32)


def plot_playback_whole_single(
    maps: list[np.ndarray],
    title: str,
    max_range: float,
    max_velocity: float,
    grid_size: int,
    min_value: float,
    max_value: float,
    aggregate: str,
    save_path: Path | None,
    no_show: bool,
) -> None:
    # Build a single map from all playback-processed frames.
    proc = [
        _playback_process_frame(
            frame=m,
            grid_size=grid_size,
            min_value=min_value,
            max_value=max_value,
        )
        for m in maps
    ]
    cube = np.stack(proc, axis=0)

    if aggregate == "mean":
        agg = np.mean(cube, axis=0)
        cb_label = "Mean Normalized Signal"
    elif aggregate == "max":
        agg = np.max(cube, axis=0)
        cb_label = "Max Normalized Signal"
    else:
        agg = np.sum(cube, axis=0)
        cb_label = "Summed Normalized Signal"

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        agg.T,
        origin="lower",
        aspect="auto",
        cmap="jet",
        extent=[0.0, max_range, -max_velocity, max_velocity],
    )
    ax.set_title(title)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Doppler [m/s]")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cb_label)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")

    if no_show:
        plt.close(fig)
    else:
        plt.show()


def resolve_session(session_arg: str, root: Path, csvs: list[Path]) -> Path:
    if session_arg.isdigit():
        idx = int(session_arg)
        if idx < 1 or idx > len(csvs):
            raise ValueError(f"Session index out of range: {idx}")
        return csvs[idx - 1]

    p = Path(session_arg).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    if p.is_dir():
        p = p / "range_doppler.csv"
    if p.exists():
        return p

    # Fallback: match session by relative path text (exact or contains, case-insensitive).
    session_text = session_arg.strip().lower().replace("\\", "/")
    rel_map = [str(csv.parent.relative_to(root)).replace("\\", "/") for csv in csvs]

    exact = [csvs[i] for i, rel in enumerate(rel_map) if rel.lower() == session_text]
    if len(exact) == 1:
        return exact[0]

    partial = [csvs[i] for i, rel in enumerate(rel_map) if session_text in rel.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        print("Multiple session matches found:")
        for pth in partial[:20]:
            print("-", pth.parent.relative_to(root))
        raise ValueError("Session selector is ambiguous. Use index from --list or full path.")

    raise ValueError(f"Session not found: {session_arg}")


def resolve_activity(activity_arg: str, activity_map: dict[str, list[Path]]) -> str:
    names = sorted(activity_map.keys())
    if activity_arg.isdigit():
        idx = int(activity_arg)
        if idx < 1 or idx > len(names):
            raise ValueError(f"Activity index out of range: {idx}")
        return names[idx - 1]

    key = activity_arg.strip().lower()
    exact = [n for n in names if n.lower() == key]
    if len(exact) == 1:
        return exact[0]

    partial = [n for n in names if key in n.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        print("Multiple activity matches found:")
        for n in partial:
            print("-", n)
        raise ValueError("Activity selector is ambiguous. Use index from --list or full name.")

    raise ValueError(f"Activity not found: {activity_arg}")


def pick_session_interactive(root: Path, csvs: list[Path]) -> Path:
    print("Pick one session by number:")
    for i, p in enumerate(csvs, start=1):
        rel = p.parent.relative_to(root)
        print(f"{i:3d}. {rel}")
    raw = input("Session number: ").strip()
    return resolve_session(raw, root, csvs)


def pick_activity_interactive(activity_map: dict[str, list[Path]]) -> str:
    names = sorted(activity_map.keys())
    print("Pick one activity by number:")
    for i, name in enumerate(names, start=1):
        print(f"{i:3d}. {name} ({len(activity_map[name])} files)")
    raw = input("Activity number: ").strip()
    return resolve_activity(raw, activity_map)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-plot session/activity radar visualizer")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "Research Data")
    parser.add_argument("--scope", choices=["session", "activity"], default="session")
    parser.add_argument("--view", choices=["all-frames", "sum", "playback-whole"], default="playback-whole")
    parser.add_argument("--session", type=str, default=None, help="session index or path")
    parser.add_argument("--activity", type=str, default=None, help="activity name")
    parser.add_argument("--pick", action="store_true", help="force interactive pick session/activity")
    parser.add_argument("--no-interactive", action="store_true", help="disable interactive prompts")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--max-range", type=float, default=14.0)
    parser.add_argument("--max-velocity", type=float, default=2.78)
    parser.add_argument("--grid-size", type=int, default=250, help="Interpolation grid size (playback style)")
    parser.add_argument("--min-value", type=float, default=2048.0, help="Playback min normalization")
    parser.add_argument("--max-value", type=float, default=4096.0, help="Playback max normalization")
    parser.add_argument(
        "--aggregate",
        choices=["sum", "mean", "max"],
        default="mean",
        help="Reducer for playback-whole view across frames",
    )
    parser.add_argument("--keep-noisy", action="store_true", help="keep corrupted/noisy frames")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    csvs = find_csvs(root)
    if not csvs:
        raise FileNotFoundError(f"No range_doppler.csv found under {root}")

    activity_map: dict[str, list[Path]] = {}
    for p in csvs:
        activity_map.setdefault(activity_name(p), []).append(p)

    if args.list:
        print("Sessions:")
        for i, p in enumerate(csvs, start=1):
            rel = p.parent.relative_to(root)
            print(f"{i:3d}. {rel}")
        print("\nActivities:")
        for i, a in enumerate(sorted(activity_map), start=1):
            print(f"{i:3d}. {a} ({len(activity_map[a])} files)")
        if args.session is None and args.activity is None:
            return

    interactive_ok = sys.stdin.isatty() and (not args.no_interactive)

    if args.scope == "session":
        if args.session is not None:
            target = resolve_session(args.session, root, csvs)
        elif args.pick or interactive_ok:
            target = pick_session_interactive(root, csvs)
        else:
            target = csvs[0]
            print("No --session provided and interactive input unavailable; using first session.")
        selected = [target]
        title = activity_name(target)
    else:
        if args.activity is not None:
            a = resolve_activity(args.activity, activity_map)
        elif args.pick or interactive_ok:
            a = pick_activity_interactive(activity_map)
        else:
            a = sorted(activity_map.keys())[0]
            print("No --activity provided and interactive input unavailable; using first activity.")
        selected = activity_map[a]
        title = a

    print(f"Scope: {args.scope}")
    print(f"View: {args.view}")
    print(f"Files selected: {len(selected)}")
    for p in selected:
        print("Loading:", p)

    maps, frame_nums, _ = collect_maps(selected)
    print(f"Total frames before filtering: {len(maps)}")

    removed = []
    if not args.keep_noisy:
        maps, frame_nums, removed = remove_noisy_maps(maps, frame_nums)
    print(f"Frames used for plot: {len(maps)}")
    print(f"Removed noisy frames: {len(removed)}")
    if removed:
        preview = removed[:20]
        suffix = "..." if len(removed) > 20 else ""
        print(f"Removed frame numbers: {preview}{suffix}")

    if args.view == "sum":
        plot_sum_rd(
            maps=maps,
            title=title,
            max_range=args.max_range,
            max_velocity=args.max_velocity,
            save_path=args.save,
            no_show=args.no_show,
        )
    elif args.view == "all-frames":
        plot_all_frames_single(
            maps=maps,
            title=title,
            max_range=args.max_range,
            save_path=args.save,
            no_show=args.no_show,
        )
    else:
        plot_playback_whole_single(
            maps=maps,
            title=title,
            max_range=args.max_range,
            max_velocity=args.max_velocity,
            grid_size=args.grid_size,
            min_value=args.min_value,
            max_value=args.max_value,
            aggregate=args.aggregate,
            save_path=args.save,
            no_show=args.no_show,
        )


if __name__ == "__main__":
    main()

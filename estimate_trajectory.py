import os
import argparse
import math
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


def load_bbox_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = set(df.columns)

    if {"frame_id", "x_min", "y_min", "x_max", "y_max"}.issubset(cols):
        df = df.rename(columns={
            "frame_id": "frame",
            "x_min": "x1",
            "y_min": "y1",
            "x_max": "x2",
            "y_max": "y2",
        })
    elif {"frame", "x1", "y1", "x2", "y2"}.issubset(cols):
        pass
    else:
        raise ValueError(f"Unexpected bbox columns: {df.columns.tolist()}")

    df = df[["frame", "x1", "y1", "x2", "y2"]].copy()
    df["frame"] = df["frame"].astype(int)
    return df.sort_values("frame").reset_index(drop=True)


def bbox_center(row):
    x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    u = int(round((x1 + x2) / 2.0))
    v = int(round((y1 + y2) / 2.0))
    return u, v


def robust_point_from_xyz(xyz_points: np.ndarray, u: int, v: int, patch_radius: int):
    H, W, _ = xyz_points.shape
    u = int(np.clip(u, 0, W - 1))
    v = int(np.clip(v, 0, H - 1))

    if patch_radius <= 0:
        p = xyz_points[v, u].astype(np.float64)
        if not np.all(np.isfinite(p)) or np.linalg.norm(p) < 1e-6:
            return None
        return p

    r = patch_radius
    u0, u1 = max(0, u - r), min(W, u + r + 1)
    v0, v1 = max(0, v - r), min(H, v + r + 1)

    pts = xyz_points[v0:v1, u0:u1].reshape(-1, 3).astype(np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        return None

    norms = np.linalg.norm(pts, axis=1)
    pts = pts[norms > 1e-6]
    if pts.size == 0:
        return None

    return np.median(pts, axis=0)


def best_point_in_bbox(xyz_points, x1, y1, x2, y2, img_w, img_h, patch=2, grid=5):
    """
    Samples a grid inside bbox; returns closest valid 3D point.
    Useful when bbox center has invalid depth.
    """
    Hxyz, Wxyz = xyz_points.shape[:2]

    def to_xyz(u, v):
        uu = int(round(u * (Wxyz / img_w)))
        vv = int(round(v * (Hxyz / img_h)))
        uu = int(np.clip(uu, 0, Wxyz - 1))
        vv = int(np.clip(vv, 0, Hxyz - 1))
        return uu, vv

    xs = np.linspace(x1, x2, grid)
    ys = np.linspace(y1, y2, grid)

    best = None
    best_norm = float("inf")
    for u in xs:
        for v in ys:
            uu, vv = to_xyz(u, v)
            p = robust_point_from_xyz(xyz_points, uu, vv, patch_radius=patch)
            if p is None:
                continue
            n = float(np.linalg.norm(p))
            if n < best_norm:
                best = p
                best_norm = n
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="dataset")
    ap.add_argument("--bbox_csv", type=str, default="bbox_light.csv")
    ap.add_argument("--rgb_dir", type=str, default="rgb")
    ap.add_argument("--xyz_dir", type=str, default="xyz")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--patch", type=int, default=3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--use_bbox_grid", action="store_true")
    args = ap.parse_args()

    dataset_dir = args.dataset_dir
    rgb_dir = os.path.join(dataset_dir, args.rgb_dir)
    xyz_dir = os.path.join(dataset_dir, args.xyz_dir)
    csv_path = os.path.join(dataset_dir, args.bbox_csv)

    ensure_dir(args.out_dir)

    df = load_bbox_csv(csv_path)

    # load one RGB to get size
    first_frame = int(df.iloc[0]["frame"])
    img_path = os.path.join(rgb_dir, f"left{first_frame:06d}.png")
    if not os.path.exists(img_path):
        files = sorted(glob.glob(os.path.join(rgb_dir, "left*.png")))
        if not files:
            raise RuntimeError(f"No RGB images found in {rgb_dir}")
        img_path = files[0]

    img0 = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img0 is None:
        raise RuntimeError(f"Could not read RGB image: {img_path}")
    img_h, img_w = img0.shape[:2]

    frames = []
    light_xyz = []

    rows = list(df.itertuples(index=False))
    if args.max_frames > 0:
        rows = rows[:args.max_frames]

    for row in tqdm(rows, desc="Reading XYZ + bboxes"):
        fid = int(row.frame)
        xyz_path = os.path.join(xyz_dir, f"depth{fid:06d}.npz")
        if not os.path.exists(xyz_path):
            continue

        xyz_points = np.load(xyz_path)["xyz"][..., :3]
        Hxyz, Wxyz = xyz_points.shape[:2]

        if args.use_bbox_grid:
            p = best_point_in_bbox(
                xyz_points,
                float(row.x1), float(row.y1), float(row.x2), float(row.y2),
                img_w, img_h,
                patch=args.patch,
                grid=5
            )
        else:
            c = bbox_center(row)
            if c is None:
                continue
            u, v = c
            u_xyz = int(round(u * (Wxyz / img_w)))
            v_xyz = int(round(v * (Hxyz / img_h)))
            p = robust_point_from_xyz(xyz_points, u_xyz, v_xyz, patch_radius=args.patch)

        if p is None:
            continue

        frames.append(fid)
        light_xyz.append(p)

    if len(light_xyz) < 5:
        raise RuntimeError(
            "Too few valid frames.\n"
            "Fix checklist:\n"
            "1) confirm xyz filenames are depth%06d.npz and match CSV frame ids\n"
            "2) try --use_bbox_grid\n"
            "3) try larger --patch (e.g. 5)\n"
        )

    light_xyz = np.vstack(light_xyz)  # (N,3): [X(right), Y(down), Z(forward)] in most camera conventions

    # =========================
    # ✅ FIX: use ground-plane axes correctly
    # forward = Z
    # lateral (left+) = -X
    # =========================
    forward = light_xyz[:, 2].copy()
    lateral = (-light_xyz[:, 0]).copy()
    light_ground = np.stack([forward, lateral], axis=1)  # (N,2) = [FWD, LAT]

    # Car position in traffic-light frame is negative of (light position in car/cam frame)
    car_light = -light_ground  # (N,2)

    # Rotate so initial direction aligns with +Forward axis
    car0 = car_light[0]
    theta0 = math.atan2(car0[1], car0[0])  # atan2(lat, fwd)
    R0 = rot2d(-theta0)

    car_world = (R0 @ car_light.T).T  # (N,2)

    # OPTIONAL: make it match example (end near origin)
    # If car is approaching the traffic light, ending closer to origin is nice visually.
    # We shift so final point is at origin (does not change shape).
    car_world = car_world - car_world[-1]

    # For plotting like the example:
    # x-axis = Forward, y-axis = Lateral  (same as example labels)
    x_plot = car_world[:, 0]
    y_plot = car_world[:, 1]

    # =========================
    # Save PNG
    # =========================
    png_path = os.path.join(args.out_dir, "partA_trajectory.png")
    plt.figure(figsize=(9, 7))
    plt.plot(x_plot, y_plot, linewidth=2, label="Ego trajectory")
    plt.scatter([x_plot[0]], [y_plot[0]], marker="x", s=180, linewidths=3, label="Start")
    plt.scatter([x_plot[-1]], [y_plot[-1]], s=160, label="End")
    plt.scatter([0], [0], marker="*", s=220, label="Traffic light (origin)")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("Forward (X, m)")
    plt.ylabel("Lateral (Y, m)")
    plt.title("Part A: Ego trajectory (ground/world frame)")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    # =========================
    # Save MP4 (same view)
    # =========================
    mp4_path = os.path.join(args.out_dir, "partA_trajectory.mp4")
    W, H = 900, 700
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(mp4_path, fourcc, args.fps, (W, H))

    pad = 2.0
    x_min, x_max = float(x_plot.min() - pad), float(x_plot.max() + pad)
    y_min, y_max = float(y_plot.min() - pad), float(y_plot.max() + pad)

    def to_img(px, py):
        ix = int((px - x_min) / (x_max - x_min + 1e-9) * (W - 1))
        iy = int((1.0 - (py - y_min) / (y_max - y_min + 1e-9)) * (H - 1))
        ix = max(0, min(W - 1, ix))
        iy = max(0, min(H - 1, iy))
        return ix, iy

    for k in tqdm(range(len(x_plot)), desc="Writing Part A MP4"):
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)

        # grid
        for gx in np.linspace(x_min, x_max, 10):
            x0, y0 = to_img(gx, y_min)
            x1, y1 = to_img(gx, y_max)
            cv2.line(canvas, (x0, y0), (x1, y1), (235, 235, 235), 1)
        for gy in np.linspace(y_min, y_max, 8):
            x0, y0 = to_img(x_min, gy)
            x1, y1 = to_img(x_max, gy)
            cv2.line(canvas, (x0, y0), (x1, y1), (235, 235, 235), 1)

        # origin (traffic light)
        ox, oy = to_img(0.0, 0.0)
        cv2.drawMarker(canvas, (ox, oy), (0, 0, 0), markerType=cv2.MARKER_STAR, markerSize=22, thickness=2)

        # trail
        for j in range(k + 1):
            ix, iy = to_img(float(x_plot[j]), float(y_plot[j]))
            cv2.circle(canvas, (ix, iy), 2, (40, 40, 220), -1)

        # current
        cx, cy = to_img(float(x_plot[k]), float(y_plot[k]))
        cv2.circle(canvas, (cx, cy), 7, (220, 0, 0), -1)

        # start/end
        sx, sy = to_img(float(x_plot[0]), float(y_plot[0]))
        ex, ey = to_img(float(x_plot[-1]), float(y_plot[-1]))
        cv2.drawMarker(canvas, (sx, sy), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=2)
        cv2.circle(canvas, (ex, ey), 8, (0, 150, 0), -1)

        cv2.putText(canvas, f"frame={frames[k]}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        vw.write(canvas)

    vw.release()

    print("Saved Part A:")
    print(" ", png_path)
    print(" ", mp4_path)


if __name__ == "__main__":
    main()


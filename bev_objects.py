import os
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_bbox_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = set(df.columns)
    if {"frame_id", "x_min", "y_min", "x_max", "y_max"}.issubset(cols):
        df = df.rename(columns={"frame_id": "frame", "x_min": "x1", "y_min": "y1", "x_max": "x2", "y_max": "y2"})
    elif {"frame", "x1", "y1", "x2", "y2"}.issubset(cols):
        pass
    else:
        raise ValueError(f"Unexpected bbox columns: {df.columns.tolist()}")
    df = df[["frame", "x1", "y1", "x2", "y2"]].copy()
    df["frame"] = df["frame"].astype(int)
    return df.sort_values("frame").reset_index(drop=True)


def bbox_center(x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
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

    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if pts.size == 0:
        return None

    norms = np.linalg.norm(pts, axis=1)
    pts = pts[norms > 1e-6]
    if pts.size == 0:
        return None

    return np.median(pts, axis=0)


def get_orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 80, 80], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def motion_mask(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, m = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m


def mask_to_centers(mask, min_area=250):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 1e-9:
            continue
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
        centers.append((u, v, a))
    centers.sort(key=lambda t: t[2], reverse=True)
    return [(u, v) for (u, v, _) in centers]


def draw_bev_point(canvas, x_fwd, y_left, x_range, y_range, color, r=4):
    H, W, _ = canvas.shape
    xmin, xmax = x_range
    ymin, ymax = y_range

    ix = int((x_fwd - xmin) / (xmax - xmin + 1e-9) * (W - 1))
    iy = int((1.0 - (y_left - ymin) / (ymax - ymin + 1e-9)) * (H - 1))
    ix = max(0, min(W - 1, ix))
    iy = max(0, min(H - 1, iy))
    cv2.circle(canvas, (ix, iy), r, color, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="dataset")
    ap.add_argument("--bbox_csv", type=str, default="bbox_light.csv")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--light_patch", type=int, default=3)
    ap.add_argument("--obj_patch", type=int, default=2)

    # ego-frame BEV bounds
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=35.0)
    ap.add_argument("--ymin", type=float, default=-15.0)
    ap.add_argument("--ymax", type=float, default=15.0)

    ap.add_argument("--barrel_min_area", type=int, default=250)
    ap.add_argument("--motion_min_area", type=int, default=500)
    args = ap.parse_args()

    dataset_dir = args.dataset_dir
    rgb_dir = os.path.join(dataset_dir, "rgb")
    xyz_dir = os.path.join(dataset_dir, "xyz")
    csv_path = os.path.join(dataset_dir, args.bbox_csv)
    ensure_dir(args.out_dir)

    df = load_bbox_csv(csv_path)
    bbox_map = {int(r.frame): (float(r.x1), float(r.y1), float(r.x2), float(r.y2)) for r in df.itertuples(index=False)}

    # frame ids are whatever appears in csv; use that ordering
    frame_ids = df["frame"].tolist()
    if args.max_frames > 0:
        frame_ids = frame_ids[:args.max_frames]

    # need RGB dims for scaling to XYZ dims
    first = frame_ids[0]
    img0 = cv2.imread(os.path.join(rgb_dir, f"left{first:06d}.png"))
    if img0 is None:
        raise RuntimeError("Could not read first RGB image to get size.")
    img_h, img_w = img0.shape[:2]

    bev_w, bev_h = 900, 900
    x_range = (args.xmin, args.xmax)
    y_range = (args.ymin, args.ymax)

    out_mp4 = os.path.join(args.out_dir, "partB_bev.mp4")
    out_dbg = os.path.join(args.out_dir, "partB_bev_debug.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, args.fps, (bev_w, bev_h))
    vdbg = cv2.VideoWriter(out_dbg, fourcc, args.fps, (img_w, img_h))

    prev_gray = None

    for fid in tqdm(frame_ids, desc="Part B: rendering"):
        img_path = os.path.join(rgb_dir, f"left{fid:06d}.png")
        xyz_path = os.path.join(xyz_dir, f"depth{fid:06d}.npz")

        bgr = cv2.imread(img_path)
        if bgr is None or not os.path.exists(xyz_path):
            continue

        xyz_points = np.load(xyz_path)["xyz"][..., :3]
        Hxyz, Wxyz = xyz_points.shape[:2]

        def to_xyz_uv(u, v):
            uu = int(round(u * (Wxyz / img_w)))
            vv = int(round(v * (Hxyz / img_h)))
            uu = max(0, min(Wxyz - 1, uu))
            vv = max(0, min(Hxyz - 1, vv))
            return uu, vv

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # --- traffic light point ---
        light_xy = None
        if fid in bbox_map:
            x1, y1, x2, y2 = bbox_map[fid]
            c = bbox_center(x1, y1, x2, y2)
            if c is not None:
                u, v = c
                uu, vv = to_xyz_uv(u, v)
                p = robust_point_from_xyz(xyz_points, uu, vv, patch_radius=args.light_patch)
                if p is not None:
                    X, Y, Z = p
                    light_xy = (X, -Y)  # forward, left

        # --- barrels (orange) ---
        barrel_mask = get_orange_mask(bgr)
        barrel_centers = mask_to_centers(barrel_mask, min_area=args.barrel_min_area)
        barrel_xy = []
        for (u, v) in barrel_centers[:30]:
            uu, vv = to_xyz_uv(u, v)
            p = robust_point_from_xyz(xyz_points, uu, vv, patch_radius=args.obj_patch)
            if p is None:
                continue
            if p[0] < 0.2:
                continue
            barrel_xy.append((p[0], -p[1]))  # forward, left

        # --- moving objects (motion) ---
        moving_xy = []
        mmask = None
        if prev_gray is not None:
            mmask = motion_mask(prev_gray, gray)
            mot_centers = mask_to_centers(mmask, min_area=args.motion_min_area)
            for (u, v) in mot_centers[:20]:
                uu, vv = to_xyz_uv(u, v)
                p = robust_point_from_xyz(xyz_points, uu, vv, patch_radius=args.obj_patch)
                if p is None:
                    continue
                if p[0] < 0.2:
                    continue
                moving_xy.append((p[0], -p[1]))
        prev_gray = gray

        # --- BEV canvas ---
        bev = np.full((bev_h, bev_w, 3), 255, dtype=np.uint8)

        # grid
        for gx in np.linspace(x_range[0], x_range[1], 8):
            ix = int((gx - x_range[0]) / (x_range[1] - x_range[0] + 1e-9) * (bev_w - 1))
            cv2.line(bev, (ix, 0), (ix, bev_h - 1), (235, 235, 235), 1)
        for gy in np.linspace(y_range[0], y_range[1], 8):
            iy = int((1.0 - (gy - y_range[0]) / (y_range[1] - y_range[0] + 1e-9)) * (bev_h - 1))
            cv2.line(bev, (0, iy), (bev_w - 1, iy), (235, 235, 235), 1)

        # car origin
        draw_bev_point(bev, 0.0, 0.0, x_range, y_range, (0, 0, 0), r=7)
        cv2.putText(bev, f"frame={fid}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # plot light
        if light_xy is not None:
            draw_bev_point(bev, light_xy[0], light_xy[1], x_range, y_range, (0, 0, 255), r=6)

        # plot barrels
        for (xf, yl) in barrel_xy:
            if xf < x_range[0] or xf > x_range[1] or yl < y_range[0] or yl > y_range[1]:
                continue
            draw_bev_point(bev, xf, yl, x_range, y_range, (0, 140, 255), r=4)

        # plot moving objects
        for (xf, yl) in moving_xy:
            if xf < x_range[0] or xf > x_range[1] or yl < y_range[0] or yl > y_range[1]:
                continue
            draw_bev_point(bev, xf, yl, x_range, y_range, (0, 200, 0), r=4)

        vw.write(bev)

        # --- debug rgb ---
        dbg = bgr.copy()
        # barrel contours
        cnts, _ = cv2.findContours(barrel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dbg, cnts, -1, (0, 140, 255), 2)

        # light bbox
        if fid in bbox_map:
            x1, y1, x2, y2 = bbox_map[fid]
            cv2.rectangle(dbg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # motion centers
        if mmask is not None:
            mot_centers = mask_to_centers(mmask, min_area=args.motion_min_area)
            for (u, v) in mot_centers[:10]:
                cv2.circle(dbg, (u, v), 8, (0, 200, 0), 2)

        vdbg.write(dbg)

    vw.release()
    vdbg.release()

    print("Saved Part B:")
    print(" ", out_mp4)
    print(" ", out_dbg)


if __name__ == "__main__":
    main()


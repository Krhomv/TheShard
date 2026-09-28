#!/usr/bin/env python3
"""Render 3/4-view thumbnails of the Shard TPU parts into renders/.

Software-rendered (no GPU/CAD needed): parses STL, projects triangles with an
orthographic camera, shades with two directional lights + ambient, and draws
far-to-near (painter's algorithm).

Usage:
    python tools/render_thumbnails.py            # render everything
    python tools/render_thumbnails.py NAME...    # re-render specific parts
"""

import struct
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "renders"
SIZE_PX = 512
DPI = 100

BG_COLOR = "#1b1e26"
BASE_COLOR = np.array([0x7F, 0xD4, 0xFF]) / 255.0  # icy blue

# view = (azimuth_deg, elevation_deg); defaults can be overridden per part
DEFAULT_VIEW = (-60.0, 22.0)

# render name -> (stl path relative to repo root, optional view override)
MANIFEST = {
    "Shard_Antenna_Backpack": ("parts/misc/Shard_Antenna_Backpack.stl", None),
    "Shard_Arm_Bumper_Crystal": ("parts/arm_bumpers/Shard_Arm_Bumper_Crystal.stl", None),
    "Shard_Arm_Bumper_Lite": ("parts/arm_bumpers/Shard_Arm_Bumper_Lite.stl", None),
    "Shard_Cap_RX_Holder": ("parts/rx_holders/Shard_Cap_RX_Holder.stl", None),
    "Shard_Cap_TVS_Holder": ("parts/misc/Shard_Cap_TVS_Holder.stl", None),
    "Shard_Crystal_Horn": ("parts/misc/Shard_Crystal_Horn.stl", None),
    "Shard_Rear_Bumper": ("parts/misc/Shard_Rear_Bumper.stl", None),
    "Shard_RX_Holder": ("parts/rx_holders/Shard_RX_Holder.stl", None),
    "Shard_ViFly_Finder_2_Holder_Forwards": ("parts/finders/Shard_ViFly_Finder_2_Holder_Forwards.stl", None),
    "Shard_ViFly_Finder_2_Holder_Sideways": ("parts/finders/Shard_ViFly_Finder_2_Holder_Sideways.stl", None),
    "Shard_ViFly_Finder_Mini_Holder_Forwards": ("parts/finders/Shard_ViFly_Finder_Mini_Holder_Forwards.stl", None),
    "Shard_ViFly_Finder_Mini_Holder_Sideways": ("parts/finders/Shard_ViFly_Finder_Mini_Holder_Sideways.stl", None),
    "Shard_VTX_Holder_O3": ("parts/vtx/o3/Shard_VTX_Holder_O3.stl", None),
    "Shard_VTX_Holder_O4_Pro": ("parts/vtx/o4_pro/Shard_VTX_Holder_O4_Pro.stl", None),
    "Shard_XTLock": ("parts/misc/Shard_XTLock.stl", None),
    "Shard_DJI_Action_2_Mount_25deg": ("parts/action_cams/dji_action_2/Shard_DJI_Action_2_Mount_25deg.stl", None),
    # one thumb per frontend family, rendered from the 25deg variant
    "Shard_Frontend_O3": ("parts/vtx/o3/Shard_Frontend_O3_25deg.stl", None),
    "Shard_Frontend_O4_Pro": ("parts/vtx/o4_pro/Shard_Frontend_O4_Pro_25deg.stl", None),
    "Shard_Frontend_O4_Pro_Itsfpv": ("parts/vtx/o4_pro/Shard_Frontend_O4_Pro_Itsfpv_25deg.stl", None),
    "Shard_Frontend_Analog": ("parts/vtx/analog/Shard_Frontend_Analog_25deg.stl", None),
}


def load_stl(path: Path):
    """Return (N, 3, 3) float64 triangle vertices; handles binary and ASCII."""
    data = path.read_bytes()
    # A valid binary STL is exactly 84 + 50*n bytes; check that first because
    # some binary files also begin with "solid".
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            raw = np.frombuffer(data, dtype=np.uint8, count=50 * n, offset=84)
            rec = raw.reshape(n, 50)[:, 12:48].copy()
            return rec.view("<f4").reshape(n, 3, 3).astype(np.float64)

    text = data.decode("utf-8", errors="replace")
    verts = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("vertex "):
            verts.append([float(v) for v in s.split()[1:4]])
    return np.array(verts, dtype=np.float64).reshape(-1, 3, 3)


def view_matrix(azim_deg: float, elev_deg: float):
    """Camera axes (right, up, forward) as rows; forward points at the part."""
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    eye = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, eye)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    return np.vstack([right, up, eye / np.linalg.norm(eye)])


def render(stl_path: Path, out_path: Path, view=DEFAULT_VIEW):
    tris = load_stl(stl_path)

    # center + uniform scale
    lo, hi = tris.reshape(-1, 3).min(axis=0), tris.reshape(-1, 3).max(axis=0)
    tris -= (lo + hi) / 2.0
    tris /= max(hi - lo)

    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    normals = np.cross(e1, e2)
    lens = np.linalg.norm(normals, axis=1)
    keep = lens > 1e-12
    tris, normals = tris[keep], normals[keep] / lens[keep, None]

    m = view_matrix(*view)
    cam = tris @ m.T  # x=right, y=up, z=depth toward camera

    # backface culling: drop faces pointing away from the camera
    facing = normals @ m[2]
    cam = cam[facing >= 0]

    # painter's algorithm: far faces first
    order = np.argsort(cam[:, :, 2].mean(axis=1))
    cam = cam[order]

    # lighting in world space, then apply the same cull/sort mask
    light1 = np.array([-0.45, 0.35, 0.82])   # key, upper-left-front
    light2 = np.array([0.70, -0.30, 0.35])   # fill, lower-right-front
    light1 /= np.linalg.norm(light1)
    light2 /= np.linalg.norm(light2)
    n_world = normals[facing >= 0][order]
    shade = 0.30 + 0.62 * np.clip(n_world @ light1, 0, None) \
                 + 0.18 * np.clip(n_world @ light2, 0, None)
    # subtle depth cue: faces nearer the camera read slightly brighter
    z = cam[:, :, 2].mean(axis=1)
    shade *= 0.88 + 0.12 * (z - z.min()) / max(np.ptp(z), 1e-9)
    shade = np.clip(shade, 0.0, 1.0)

    rgb = np.clip(BASE_COLOR[None, :] * shade[:, None], 0, 1)
    edge_rgb = np.clip(rgb * 0.55, 0, 1)  # darker seam hides sort artifacts

    fig = plt.figure(figsize=(SIZE_PX / DPI, SIZE_PX / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG_COLOR)
    ax.set_axis_off()

    poly = PolyCollection(
        cam[:, :, :2],
        facecolors=rgb,
        edgecolors=edge_rgb,
        linewidths=0.25,
    )
    ax.add_collection(poly)

    extent = max(np.ptp(cam[:, :, 0]), np.ptp(cam[:, :, 1])) / 2.0 * 1.12
    cx = (cam[:, :, 0].max() + cam[:, :, 0].min()) / 2.0
    cy = (cam[:, :, 1].max() + cam[:, :, 1].min()) / 2.0
    ax.set_xlim(cx - extent, cx + extent)
    ax.set_ylim(cy - extent, cy + extent)
    ax.set_aspect("equal")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=BG_COLOR)
    plt.close(fig)


def main(argv):
    names = argv[1:]
    unknown = [n for n in names if n not in MANIFEST]
    if unknown:
        sys.exit(f"unknown part(s): {', '.join(unknown)}\nknown: {', '.join(sorted(MANIFEST))}")
    OUT_DIR.mkdir(exist_ok=True)
    for name in names or sorted(MANIFEST):
        rel, view = MANIFEST[name]
        stl = ROOT / rel
        out = OUT_DIR / f"{name}.png"
        render(stl, out, view or DEFAULT_VIEW)
        print(f"rendered {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv)

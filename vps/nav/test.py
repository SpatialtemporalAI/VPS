import numpy as np
import open3d as o3d
import time
import matplotlib.pyplot as plt

def fast_floor_by_z_hist_fast(points,
                              cam_z,
                              voxel_down=0.02,
                              nbins=80,
                              peak_window=0.12,
                              floor_tol=0.06,
                              voxel_filter=0.05,
                              min_voxel_points=50,
                              smooth_k=3,
                              visualize=False,
                              out_prefix=None,
                              view_params=None):
    """
    Fast floor detection (horizontal floor assumption) + optional visualization.

    Returns: floor_mask (N,), z0 (float or None), diagnostics dict
    """
    s = time.time()
    # --- downsample if requested (Open3D voxel downsample is fast) ---
    if voxel_down and voxel_down > 0:
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(points)
        pcd_tmp = pcd_tmp.voxel_down_sample(voxel_size=voxel_down)
        pts = np.asarray(pcd_tmp.points)
    else:
        pts = points

    # prepare outputs
    N = points.shape[0]
    floor_mask_full = np.zeros(N, dtype=bool)
    z0 = None
    diagnostics = {}

    # --- filter below camera ---
    z = pts[:, 2]
    below_mask_small = z <= cam_z
    if not np.any(below_mask_small):
        diagnostics['error'] = "No below-camera points"
        return floor_mask_full, None, diagnostics

    z_below = z[below_mask_small]
    zmin, zmax = float(z_below.min()), float(z_below.max())
    if zmax - zmin < 1e-9:
        z0 = float(zmin)
        floor_mask_full = np.abs(points[:, 2] - z0) <= floor_tol
        floor_mask_full &= (points[:, 2] <= cam_z)
        diagnostics['note'] = "Degenerate z range"
        diagnostics['z0'] = z0
        diagnostics['time'] = time.time() - s
        return floor_mask_full, z0, diagnostics

    # histogram + smoothing (np.convolve is fast)
    bins = np.linspace(zmin, zmax, nbins + 1)
    h, _ = np.histogram(z_below, bins=bins)
    k = max(1, int(smooth_k))
    if k > 1:
        kernel = np.ones(k, dtype=float) / k
        h_s = np.convolve(h.astype(float), kernel, mode='same')
    else:
        h_s = h.astype(float)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # choose peak among lower portion (prefer lower peaks)
    cutoff = zmin + 0.6 * (zmax - zmin)
    cand_idxs = np.where(centers <= cutoff)[0]
    if cand_idxs.size == 0:
        peak_idx = int(np.argmax(h_s))
    else:
        local = h_s[cand_idxs]
        arg = int(np.argmax(local))
        peak_idx = int(cand_idxs[arg])
    z_peak = float(centers[peak_idx])

    # candidate selection on downsampled pts
    z_vs = z
    cand_mask_vs = np.abs(z_vs - z_peak) <= peak_window
    if not np.any(cand_mask_vs):
        z0 = float(np.percentile(z_below, 7.5))
    else:
        # voxelize XY on downsampled cloud
        xy = pts[:, :2]
        mins = xy.min(axis=0)
        idxs = np.floor((xy - mins) / voxel_filter).astype(np.int64)
        ix = idxs[:, 0]; iy = idxs[:, 1]
        Ny = int(iy.max()) + 1
        linear = ix * Ny + iy
        linear_cand = linear[cand_mask_vs]
        if linear_cand.size == 0:
            z0 = float(np.median(z_vs[cand_mask_vs]))
        else:
            unique_ids, counts = np.unique(linear_cand, return_counts=True)
            valid_voxels = unique_ids[counts >= min_voxel_points]
            if valid_voxels.size == 0:
                z0 = float(np.median(z_vs[cand_mask_vs]))
            else:
                valid_mask = np.isin(linear, valid_voxels)
                sel = cand_mask_vs & valid_mask
                if np.any(sel):
                    z0 = float(np.median(z_vs[sel]))
                else:
                    z0 = float(np.median(z_vs[cand_mask_vs]))

    # final mask on original points
    floor_mask_full = np.abs(points[:, 2] - z0) <= floor_tol
    floor_mask_full &= (points[:, 2] <= cam_z)

    diagnostics.update({
        'z_peak': z_peak,
        'z0': z0,
        'n_total': int(points.shape[0]),
        'n_floor': int(np.count_nonzero(floor_mask_full)),
        'time_s': time.time() - s
    })

    # --- Visualization & outputs ---
    if visualize or (out_prefix is not None):
        # build colored point cloud (priority: above->red, below-non-floor->blue, floor->green)
        colors = np.tile(np.array([0.8, 0.8, 0.8], dtype=float), (N, 1))
        above_mask = points[:, 2] > cam_z
        colors[above_mask] = np.array([1.0, 0.0, 0.0])
        below_non_floor = (points[:, 2] <= cam_z) & (~floor_mask_full)
        colors[below_non_floor] = np.array([0.0, 0.0, 1.0])
        if np.any(floor_mask_full):
            colors[floor_mask_full] = np.array([0.0, 1.0, 0.0])

        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(points)
        pcd_out.colors = o3d.utility.Vector3dVector(colors)

        # save colored pcd if requested
        if out_prefix is not None:
            try:
                colored_path = f"{out_prefix}colored.ply"
                o3d.io.write_point_cloud(colored_path, pcd_out)
                diagnostics['colored_ply'] = colored_path
            except Exception as e:
                diagnostics['save_error'] = str(e)

        # histogram plot with markers
        try:
            plt.figure(figsize=(8,3))
            plt.plot(centers, h_s, label="smoothed hist (downsampled below)")
            plt.axvline(z_peak, color="g", linestyle="--", label=f"z_peak={z_peak:.3f}")
            # plt.axvline(z0, color="r", linestyle="-.", label=f"z0={z0:.3f}")
            plt.axvline(cam_z, color="k", linestyle=":", label=f"cam_z={cam_z:.3f}")
            plt.legend()
            plt.tight_layout()
            if out_prefix is not None:
                hist_path = f"{out_prefix}hist.png"
                plt.savefig(hist_path, dpi=150)
                diagnostics['hist_png'] = hist_path
            if visualize:
                plt.show()
            plt.close()
        except Exception as e:
            diagnostics['hist_error'] = str(e)

        # Open3D visualization
        if visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="floor segmentation (R:red, F:green, B:blue)", width=1000, height=700)
            vis.add_geometry(pcd_out)
            vc = vis.get_view_control()
            center = pcd_out.get_center()
            vc.set_lookat(center)
            # view params
            if view_params is None:
                view_params = {}
            front = np.array(view_params.get('front', [-1.0, -1.0, -0.5]), dtype=float)
            fnorm = np.linalg.norm(front)
            if fnorm == 0:
                front = np.array([0.0, 0.0, -1.0])
            else:
                front = front / fnorm
            vc.set_front(front.tolist())
            vc.set_up(view_params.get('up', [0.0, 0.0, 1.0]))
            try:
                vc.set_zoom(view_params.get('zoom', 0.7))
            except Exception:
                pass
            vis.run()
            vis.destroy_window()

    return floor_mask_full, z0, diagnostics


# ------------------ example use ------------------
if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("/home/phw/visual-localization/VPS/output.ply")
    pts = np.asarray(pcd.points)
    for i in range(3):
        mask, z0, info = fast_floor_by_z_hist_fast(pts, 0.69,
                                                   voxel_down=0.02,
                                                   nbins=80,
                                                   peak_window=0.12,
                                                   floor_tol=0.15,
                                                   voxel_filter=0.05,
                                                   min_voxel_points=50,
                                                   smooth_k=3,
                                                   visualize=True,
                                                   out_prefix="test_out_")
        print("run", i, "z0=", z0, "n_floor=", int(np.count_nonzero(mask)), "time_s=", info['time_s'])

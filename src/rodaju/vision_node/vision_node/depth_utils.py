#!/usr/bin/env python3
"""Depth 이미지 기반 3D 좌표 변환 유틸리티 (free function)."""


def pixel_to_3d(
    cx: int, cy: int,
    depth_frame,
    intrinsics: dict,
    depth_scale: float = 0.001,
    roi_radius: int = 3,
    logger=None,
):
    """픽셀 좌표 + ROI depth 중앙값 → 3D 좌표 (m).

    Args:
        cx, cy:       픽셀 좌표
        depth_frame:  numpy depth 이미지 (HxW, uint16 또는 float)
        intrinsics:   {"fx", "fy", "ppx", "ppy"}
        depth_scale:  depth 단위 → m 변환 계수 (기본 0.001, mm→m)
        roi_radius:   ROI 반경 (픽셀)
        logger:       rclpy logger (optional)

    Returns:
        (x_m, y_m, z_m)
    """
    if depth_frame is None or intrinsics is None:
        return 0.0, 0.0, 0.0
    try:
        import numpy as np
        h, w = depth_frame.shape[:2]
        r    = roi_radius
        x1   = max(0, cx - r); x2 = min(w, cx + r + 1)
        y1   = max(0, cy - r); y2 = min(h, cy + r + 1)
        roi  = depth_frame[y1:y2, x1:x2].astype(float)
        valid = roi[roi > 0]
        if valid.size == 0:
            return 0.0, 0.0, 0.0

        z_m = float(np.median(valid)) * depth_scale
        if z_m < 0.01:
            return 0.0, 0.0, 0.0

        fx, fy   = intrinsics["fx"],  intrinsics["fy"]
        ppx, ppy = intrinsics["ppx"], intrinsics["ppy"]
        x_m = (cx - ppx) * z_m / fx
        y_m = (cy - ppy) * z_m / fy
        return x_m, y_m, z_m
    except Exception as e:
        if logger:
            logger.warn(f"3D transform error: {e}")
        return 0.0, 0.0, 0.0


def obb_to_3d(
    points_px,
    depth_frame,
    intrinsics: dict,
    depth_scale: float = 0.001,
    n_slices: int = 12,
    logger=None,
):
    """OBB 장축 슬라이싱 → 그립 포인트 3D 좌표 계산.

    1. OBB 장축 방향으로 n_slices 등분
    2. 각 슬라이스에서 depth 최솟값(물체 최상단) 픽셀 추출
    3. 그 픽셀들의 픽셀 XY 중앙값 → grasp_xy
    4. grasp_z = (top_z + bottom_z) / 2

    Args:
        points_px:  OBB 4꼭짓점 (4,2) int32 numpy array
        depth_frame: numpy depth 이미지
        intrinsics:  {"fx", "fy", "ppx", "ppy"}
        depth_scale: depth 단위 → m 변환 계수
        n_slices:   슬라이스 수
        logger:     rclpy logger (optional)

    Returns:
        (x_m, y_m, grasp_z_m, cx_px, cy_px)
    """
    if depth_frame is None or intrinsics is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        import numpy as np
        import cv2

        h, w     = depth_frame.shape[:2]
        fx, fy   = intrinsics["fx"],  intrinsics["fy"]
        ppx, ppy = intrinsics["ppx"], intrinsics["ppy"]

        # OBB 폴리곤 마스크 생성
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points_px], 255)

        # 마스크 내 유효 depth 픽셀 추출
        ys, xs = np.where((mask > 0) & (depth_frame > 0))
        if len(xs) < n_slices:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        depths = depth_frame[ys, xs].astype(float) * depth_scale
        valid  = depths > 0.01
        xs, ys, depths = xs[valid], ys[valid], depths[valid]
        if len(xs) < n_slices:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # ── OBB 장축 방향 계산 ──────────────────────────────
        pts = points_px.astype(float)
        d0 = pts[1] - pts[0]
        d1 = pts[2] - pts[1]
        long_axis = d0 if np.linalg.norm(d0) >= np.linalg.norm(d1) else d1
        long_axis = long_axis / np.linalg.norm(long_axis)

        cx_obb = float(np.mean(pts[:, 0]))
        cy_obb = float(np.mean(pts[:, 1]))

        pts_arr  = np.stack([xs, ys], axis=1).astype(float)
        centered = pts_arr - np.array([cx_obb, cy_obb])
        proj     = centered @ long_axis

        proj_min, proj_max = proj.min(), proj.max()
        edges = np.linspace(proj_min, proj_max, n_slices + 1)

        top_xs, top_ys, top_depths = [], [], []
        slice_all_xs, slice_all_ys = [], []
        for k in range(n_slices):
            in_slice = (proj >= edges[k]) & (proj < edges[k + 1])
            if not np.any(in_slice):
                continue
            slice_depths = depths[in_slice]
            best         = np.argmin(slice_depths)
            top_xs.append(xs[in_slice][best])
            top_ys.append(ys[in_slice][best])
            top_depths.append(slice_depths[best])
            slice_all_xs.append(xs[in_slice])
            slice_all_ys.append(ys[in_slice])

        if len(top_depths) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        top_depths = np.array(top_depths)
        RANK = min(2, len(top_depths) - 1)
        best_slice = int(np.argsort(top_depths)[RANK])
        top_z    = float(top_depths[best_slice])
        bottom_z = float(np.median(
            depths[np.argsort(depths)[-max(1, int(len(depths) * 0.15)):]]
        ))
        grasp_z = (top_z + bottom_z) / 2.0

        cx_px = float(np.median(slice_all_xs[best_slice]))
        cy_px = float(np.median(slice_all_ys[best_slice]))
        x_m   = (cx_px - ppx) * grasp_z / fx
        y_m   = (cy_px - ppy) * grasp_z / fy

        if logger:
            logger.debug(
                f"[OBB3D] slices={len(top_depths)} top_z={top_z:.3f} "
                f"bottom_z={bottom_z:.3f} grasp_z={grasp_z:.3f}"
            )
        return x_m, y_m, grasp_z, cx_px, cy_px

    except Exception as e:
        if logger:
            logger.warn(f"OBB 3D error: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0

#!/usr/bin/env python3
"""Build AnyRecon condition videos from video_to_world Stage 0 outputs."""

import argparse
import json
import math
import os

import cv2
import numpy as np

from run_pi3 import (
    center_crop_to_size,
    ensure_dir,
    render_point_cloud_to_memory,
    save_video,
)


OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _as_homogeneous(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        output = np.eye(4, dtype=np.float32)
        output[:3, :4] = matrix
        return output
    raise ValueError(f"Expected a (3, 4) or (4, 4) matrix, got {matrix.shape}.")


def load_stage0_results(npz_path):
    pi3_raw_keys = {
        "image",
        "points",
        "conf",
        "camera_poses",
        "intrinsics",
    }
    depth_keys = {"depth", "conf", "extrinsics", "intrinsics", "image"}

    with np.load(npz_path) as data:
        available = set(data.files)
        if pi3_raw_keys.issubset(available):
            results = {
                key: np.asarray(data[key])
                for key in pi3_raw_keys
            }
            results["_geometry_type"] = "pi3_raw"
        elif depth_keys.issubset(available):
            results = {
                key: np.asarray(data[key])
                for key in depth_keys
            }
            results["_geometry_type"] = "depth"
        else:
            raise KeyError(
                f"Unsupported NPZ layout in {npz_path}. "
                f"Available keys: {sorted(available)}. Expected either "
                f"Pi3 raw keys {sorted(pi3_raw_keys)} or "
                f"depth keys {sorted(depth_keys)}."
            )

    images = results["image"]
    conf = results["conf"]

    if results["_geometry_type"] == "pi3_raw":
        points = results["points"]
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError(
                f"points must have shape (N, H, W, 3), got {points.shape}."
            )
        n, h, w, _ = points.shape
        if conf.shape != (n, h, w):
            raise ValueError(
                f"conf has shape {conf.shape}, expected {(n, h, w)}."
            )
        if images.shape != (n, h, w, 3):
            raise ValueError(
                f"image has shape {images.shape}, expected {(n, h, w, 3)}."
            )
        if results["camera_poses"].shape != (n, 4, 4):
            raise ValueError(
                "camera_poses must have shape "
                f"{(n, 4, 4)}, got {results['camera_poses'].shape}."
            )
        if results["intrinsics"].shape != (n, 3, 3):
            raise ValueError(
                "intrinsics must have shape "
                f"{(n, 3, 3)}, got {results['intrinsics'].shape}."
            )
        print("Detected Pi3 raw NPZ; using saved global point maps directly.")
        return results

    depth = results["depth"]
    extrinsics = results["extrinsics"]
    intrinsics = results["intrinsics"]

    if depth.ndim != 3:
        raise ValueError(f"depth must have shape (N, H, W), got {depth.shape}.")
    n, h, w = depth.shape
    expected_shapes = {
        "conf": (n, h, w),
        "extrinsics": [(n, 3, 4), (n, 4, 4)],
        "intrinsics": (n, 3, 3),
        "image": (n, h, w, 3),
    }
    if conf.shape != expected_shapes["conf"]:
        raise ValueError(f"conf has shape {conf.shape}, expected {(n, h, w)}.")
    if extrinsics.shape not in expected_shapes["extrinsics"]:
        raise ValueError(
            f"extrinsics has shape {extrinsics.shape}, expected "
            f"{expected_shapes['extrinsics']}."
        )
    if intrinsics.shape != expected_shapes["intrinsics"]:
        raise ValueError(
            f"intrinsics has shape {intrinsics.shape}, expected {(n, 3, 3)}."
        )
    if images.shape != expected_shapes["image"]:
        raise ValueError(
            f"image has shape {images.shape}, expected {(n, h, w, 3)}."
        )

    print("Detected depth-based Stage 0 NPZ; back-projecting depth maps.")
    return results


def build_pi3_raw_point_cloud(
    results,
    point_frame_indices,
    conf_threshold,
    pixel_stride,
    max_points,
    seed,
):
    """Match run_pi3.py point selection using saved Pi3 global point maps."""
    if pixel_stride < 1:
        raise ValueError("--pixel_stride must be at least 1.")

    points_map = results["points"]
    conf = results["conf"]
    images = results["image"]

    all_points = []
    all_colors = []
    all_source_indices = []

    for frame_idx in point_frame_indices:
        points_i = points_map[
            frame_idx,
            ::pixel_stride,
            ::pixel_stride,
        ]
        conf_i = conf[
            frame_idx,
            ::pixel_stride,
            ::pixel_stride,
        ]
        image_i = images[
            frame_idx,
            ::pixel_stride,
            ::pixel_stride,
        ]

        # preprocess_video_pi3.py stores sigmoid(conf) after multiplying
        # by the non-edge mask. Applying > 0.1 therefore reproduces:
        # (sigmoid(conf) > 0.1) & non_edge.
        valid = conf_i > conf_threshold
        if not np.any(valid):
            print(f"Warning: frame {frame_idx} contributed no valid points.")
            continue

        points = points_i[valid].astype(np.float32)
        colors = image_i[valid]
        if colors.dtype != np.uint8:
            colors_float = colors.astype(np.float32)
            if colors_float.max(initial=0.0) <= 1.0:
                colors_float *= 255.0
            colors = np.clip(colors_float, 0, 255).astype(np.uint8)

        all_points.append(points)
        all_colors.append(colors)
        all_source_indices.append(
            np.full(points.shape[0], frame_idx, dtype=np.int32)
        )

    if not all_points:
        raise ValueError("Selected Pi3 frames produced an empty point cloud.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    source_indices = np.concatenate(all_source_indices, axis=0)

    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
        source_indices = source_indices[keep]

    print(
        f"Selected {len(points):,} saved Pi3 points from "
        f"{len(point_frame_indices)} frames with conf > {conf_threshold}."
    )
    return points, colors, source_indices


def build_stage0_point_cloud(
    results,
    point_frame_indices,
    conf_percentile,
    conf_threshold,
    pixel_stride,
    max_points,
    seed,
):
    if results["_geometry_type"] == "pi3_raw":
        return build_pi3_raw_point_cloud(
            results=results,
            point_frame_indices=point_frame_indices,
            conf_threshold=conf_threshold,
            pixel_stride=pixel_stride,
            max_points=max_points,
            seed=seed,
        )

    depth = results["depth"]
    conf = results["conf"]
    extrinsics = results["extrinsics"]
    intrinsics = results["intrinsics"]
    images = results["image"]

    if not 0.0 <= conf_percentile <= 100.0:
        raise ValueError("--conf_percentile must be in [0, 100].")
    if pixel_stride < 1:
        raise ValueError("--pixel_stride must be at least 1.")

    confidence_values = []
    for frame_idx in point_frame_indices:
        depth_i = depth[frame_idx, ::pixel_stride, ::pixel_stride]
        conf_i = conf[frame_idx, ::pixel_stride, ::pixel_stride]
        valid_depth = np.isfinite(depth_i) & (depth_i > 0)
        if np.any(valid_depth):
            confidence_values.append(conf_i[valid_depth])

    if not confidence_values:
        raise ValueError("No valid positive depth values found in selected frames.")

    conf_threshold = float(
        np.percentile(np.concatenate(confidence_values), conf_percentile)
    )
    print(
        f"Using global confidence threshold {conf_threshold:.6f} "
        f"at percentile {conf_percentile:.1f}."
    )

    height, width = depth.shape[1:]
    v_grid, u_grid = np.mgrid[
        0:height:pixel_stride,
        0:width:pixel_stride,
    ]

    all_points = []
    all_colors = []
    all_source_indices = []

    for frame_idx in point_frame_indices:
        depth_i = depth[frame_idx, ::pixel_stride, ::pixel_stride]
        conf_i = conf[frame_idx, ::pixel_stride, ::pixel_stride]
        image_i = images[frame_idx, ::pixel_stride, ::pixel_stride]

        valid = (
            np.isfinite(depth_i)
            & (depth_i > 0)
            & np.isfinite(conf_i)
            & (conf_i >= conf_threshold)
        )
        if not np.any(valid):
            print(f"Warning: frame {frame_idx} contributed no valid points.")
            continue

        pixels = np.stack(
            [
                u_grid[valid],
                v_grid[valid],
                np.ones(np.count_nonzero(valid), dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)

        rays_camera = (
            np.linalg.inv(intrinsics[frame_idx]).astype(np.float32)
            @ pixels.T
        ).T
        points_camera = rays_camera * depth_i[valid, None].astype(np.float32)

        c2w = np.linalg.inv(_as_homogeneous(extrinsics[frame_idx]))
        points_world = (
            c2w[:3, :3] @ points_camera.T
        ).T + c2w[:3, 3]

        colors = image_i[valid]
        if colors.dtype != np.uint8:
            colors_float = colors.astype(np.float32)
            if colors_float.max(initial=0.0) <= 1.0:
                colors_float *= 255.0
            colors = np.clip(colors_float, 0, 255).astype(np.uint8)

        all_points.append(points_world.astype(np.float32))
        all_colors.append(colors)
        all_source_indices.append(
            np.full(points_world.shape[0], frame_idx, dtype=np.int32)
        )

    if not all_points:
        raise ValueError("Selected Stage 0 frames produced an empty point cloud.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    source_indices = np.concatenate(all_source_indices, axis=0)

    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
        source_indices = source_indices[keep]

    print(
        f"Built Stage 0 point cloud with {len(points):,} points "
        f"from {len(point_frame_indices)} frames."
    )
    return points, colors, source_indices


def load_v2w_trajectory(
    transforms_path,
    trajectory_start,
    num_render_frames,
    render_width,
    render_height,
):
    with open(transforms_path, "r") as file:
        data = json.load(file)

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames found in {transforms_path}.")

    if trajectory_start < 0 or trajectory_start >= len(frames):
        raise ValueError(
            f"--trajectory_start must be in [0, {len(frames) - 1}]."
        )

    selected_frames = frames[trajectory_start:]
    if num_render_frames > 0:
        selected_frames = selected_frames[:num_render_frames]
    if not selected_frames:
        raise ValueError("The selected trajectory is empty.")

    json_width = int(data["w"])
    json_height = int(data["h"])
    output_width = json_width if render_width <= 0 else render_width
    output_height = json_height if render_height <= 0 else render_height

    scale_x = float(output_width) / float(json_width)
    scale_y = float(output_height) / float(json_height)
    intrinsic = np.array(
        [
            [float(data["fl_x"]) * scale_x, 0.0, float(data["cx"]) * scale_x],
            [0.0, float(data["fl_y"]) * scale_y, float(data["cy"]) * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    scale_factor = float(data.get("scale_factor", 1.0))
    extrinsics_w2c = []
    poses_c2w = []
    for frame in selected_frames:
        c2w_opengl = np.asarray(frame["transform_matrix"], dtype=np.float32)
        if c2w_opengl.shape != (4, 4):
            raise ValueError(
                f"transform_matrix must have shape (4, 4), "
                f"got {c2w_opengl.shape}."
            )
        if not np.isfinite(c2w_opengl).all():
            raise ValueError("Trajectory contains NaN or infinite values.")

        c2w_opencv = c2w_opengl @ OPENGL_TO_OPENCV
        c2w_opencv[:3, 3] *= scale_factor
        poses_c2w.append(c2w_opencv)
        extrinsics_w2c.append(np.linalg.inv(c2w_opencv))

    extrinsics_w2c = np.stack(extrinsics_w2c).astype(np.float32)
    poses_c2w = np.stack(poses_c2w).astype(np.float32)
    intrinsics = np.repeat(
        intrinsic[None],
        len(extrinsics_w2c),
        axis=0,
    )

    print(
        f"Loaded {len(extrinsics_w2c)} trajectory poses from "
        f"{transforms_path} at {output_width}x{output_height}; "
        f"scale_factor={scale_factor}."
    )
    return (
        extrinsics_w2c,
        poses_c2w,
        intrinsics,
        output_width,
        output_height,
    )


def choose_condition_frames(
    use_retrieval,
    default_condition_indices,
    rendered_source_stats,
    top_k,
):
    if not use_retrieval:
        return list(default_condition_indices)

    aggregated = {}
    for stats in rendered_source_stats:
        for source_idx, count in stats.items():
            aggregated[int(source_idx)] = (
                aggregated.get(int(source_idx), 0) + int(count)
            )

    ranked = sorted(
        aggregated.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    selected = sorted(source_idx for source_idx, _ in ranked[:top_k])
    if not selected:
        return list(default_condition_indices)
    return selected


def save_bridge_outputs(
    results,
    points,
    colors,
    source_indices,
    extrinsics_w2c,
    poses_c2w,
    intrinsics,
    render_width,
    render_height,
    args,
):
    condition_dir = os.path.join(args.output_dir, "condition")
    mask_dir = os.path.join(args.output_dir, "mask")
    camera_dir = os.path.join(args.output_dir, "cameras")
    ensure_dir(condition_dir)
    ensure_dir(mask_dir)
    ensure_dir(camera_dir)

    num_trajectory_frames = len(extrinsics_w2c)
    num_chunks = math.ceil(num_trajectory_frames / args.chunk_size)
    remainder = num_trajectory_frames % args.chunk_size
    if remainder and not args.allow_partial_chunk:
        usable_frames = num_trajectory_frames - remainder
        print(
            f"Warning: dropping the final {remainder} trajectory frames because "
            f"AnyRecon chunks use {args.chunk_size} render frames. "
            "Pass --allow_partial_chunk to keep them."
        )
        num_chunks = usable_frames // args.chunk_size

    if num_chunks == 0:
        raise ValueError(
            f"Need at least {args.chunk_size} trajectory frames, or pass "
            "--allow_partial_chunk."
        )

    final_size = (args.final_width, args.final_height)
    white_mask = np.full(
        (args.final_height, args.final_width, 3),
        255,
        dtype=np.uint8,
    )
    images_rgb = results["image"]
    metadata = {
        "stage0_npz": os.path.abspath(args.stage0_npz),
        "trajectory_json": os.path.abspath(args.trajectory_json),
        "coordinate_convention": "OpenCV c2w/w2c after OpenGL conversion",
        "render_width": render_width,
        "render_height": render_height,
        "trajectory_start": args.trajectory_start,
        "frames": [],
    }

    for chunk_idx in range(num_chunks):
        start = chunk_idx * args.chunk_size
        end = min(start + args.chunk_size, num_trajectory_frames)
        if end - start < args.chunk_size and not args.allow_partial_chunk:
            break

        rendered_frames, rendered_masks, rendered_source_stats = (
            render_point_cloud_to_memory(
                points_3d=points,
                points_rgb=colors,
                extrinsics=extrinsics_w2c[start:end],
                intrinsics=intrinsics[start:end],
                image_size=(render_width, render_height),
                points_source_indices=source_indices,
            )
        )

        selected_conditions = choose_condition_frames(
            use_retrieval=args.use_retrieval,
            default_condition_indices=range(args.num_cond_frames),
            rendered_source_stats=rendered_source_stats,
            top_k=args.top_k_condition_frames,
        )

        condition_video = []
        mask_video = []
        for frame_idx in selected_conditions:
            capture_bgr = cv2.cvtColor(
                images_rgb[frame_idx],
                cv2.COLOR_RGB2BGR,
            )
            condition_video.append(
                center_crop_to_size(
                    capture_bgr,
                    args.final_width,
                    args.final_height,
                )
            )
            mask_video.append(white_mask)

        for rendered_frame, rendered_mask in zip(
            rendered_frames,
            rendered_masks,
        ):
            condition_video.append(
                center_crop_to_size(
                    rendered_frame,
                    args.final_width,
                    args.final_height,
                )
            )
            mask_video.append(
                center_crop_to_size(
                    rendered_mask,
                    args.final_width,
                    args.final_height,
                )
            )

        filename = (
            f"{args.scene_name}_chunk_{chunk_idx:04d}_"
            f"frames_{start}_{end - 1}"
        )
        condition_path = os.path.join(condition_dir, f"{filename}.mp4")
        mask_path = os.path.join(mask_dir, f"{filename}.mp4")
        if not save_video(condition_video, condition_path, fps=args.fps):
            raise RuntimeError(f"Failed to save {condition_path}.")
        if not save_video(mask_video, mask_path, fps=args.fps):
            raise RuntimeError(f"Failed to save {mask_path}.")

        info_path = os.path.join(condition_dir, f"{filename}_info.txt")
        with open(info_path, "w") as file:
            file.write(f"Condition Frame Count: {len(selected_conditions)}\n")
            file.write(f"Render Frame Count: {end - start}\n")
            file.write(f"Condition Frame Indices: {selected_conditions}\n")

        for local_idx in range(end - start):
            global_idx = start + local_idx
            metadata["frames"].append(
                {
                    "frame_id": global_idx,
                    "chunk_id": chunk_idx,
                    "transform_matrix_c2w_opencv": (
                        poses_c2w[global_idx].tolist()
                    ),
                    "extrinsic_w2c_opencv": (
                        extrinsics_w2c[global_idx].tolist()
                    ),
                    "intrinsic": intrinsics[global_idx].tolist(),
                    "source_frames": sorted(
                        int(key)
                        for key in rendered_source_stats[local_idx].keys()
                    ),
                }
            )

        print(
            f"Saved chunk {chunk_idx + 1}/{num_chunks}: "
            f"{condition_path}"
        )

    metadata_path = os.path.join(
        camera_dir,
        f"{args.scene_name}_v2w_bridge.json",
    )
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render video_to_world Stage 0 geometry along its exported "
            "extended trajectory and package it for AnyRecon."
        )
    )
    parser.add_argument(
        "--stage0_npz",
        required=True,
        help=(
            "Path to results_pi3_raw.npz (exact Pi3 points) or "
            "depth-based results.npz."
        ),
    )
    parser.add_argument(
        "--trajectory_json",
        required=True,
        help="Path to gs_video/0000_extend_transforms.json.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_name", default=None)
    parser.add_argument(
        "--num_cond_frames",
        type=int,
        default=6,
        help="Leading Stage 0 images supplied as default AnyRecon conditions.",
    )
    parser.add_argument(
        "--use_all_stage0_points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build geometry from every Stage 0 frame instead of condition frames only.",
    )
    parser.add_argument(
        "--conf_percentile",
        type=float,
        default=40.0,
        help="Global confidence percentile for depth-based NPZ inputs.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.1,
        help="Absolute confidence threshold for Pi3 raw NPZ inputs.",
    )
    parser.add_argument("--pixel_stride", type=int, default=1)
    parser.add_argument("--max_points", type=int, default=1_000_000)
    parser.add_argument("--trajectory_start", type=int, default=0)
    parser.add_argument(
        "--num_render_frames",
        type=int,
        default=35,
        help="Number of trajectory poses to render; use -1 for all.",
    )
    parser.add_argument("--render_width", type=int, default=-1)
    parser.add_argument("--render_height", type=int, default=-1)
    parser.add_argument("--final_width", type=int, default=896)
    parser.add_argument("--final_height", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=35)
    parser.add_argument(
        "--allow_partial_chunk",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use_retrieval",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--top_k_condition_frames", type=int, default=10)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.stage0_npz = os.path.abspath(args.stage0_npz)
    args.trajectory_json = os.path.abspath(args.trajectory_json)
    args.output_dir = os.path.abspath(args.output_dir)

    if args.scene_name is None:
        scene_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(args.stage0_npz)
            )
        )
        args.scene_name = os.path.basename(scene_root) or "scene"

    if args.chunk_size < 1:
        raise ValueError("--chunk_size must be at least 1.")
    if args.num_cond_frames < 1:
        raise ValueError("--num_cond_frames must be at least 1.")

    results = load_stage0_results(args.stage0_npz)
    num_stage0_frames = results["image"].shape[0]
    if args.num_cond_frames > num_stage0_frames:
        raise ValueError(
            f"--num_cond_frames={args.num_cond_frames} exceeds the "
            f"{num_stage0_frames} Stage 0 frames."
        )

    if args.use_all_stage0_points:
        point_frame_indices = list(range(num_stage0_frames))
    else:
        point_frame_indices = list(range(args.num_cond_frames))

    points, colors, source_indices = build_stage0_point_cloud(
        results=results,
        point_frame_indices=point_frame_indices,
        conf_percentile=args.conf_percentile,
        conf_threshold=args.conf_threshold,
        pixel_stride=args.pixel_stride,
        max_points=args.max_points,
        seed=args.seed,
    )

    (
        extrinsics_w2c,
        poses_c2w,
        intrinsics,
        render_width,
        render_height,
    ) = load_v2w_trajectory(
        transforms_path=args.trajectory_json,
        trajectory_start=args.trajectory_start,
        num_render_frames=args.num_render_frames,
        render_width=args.render_width,
        render_height=args.render_height,
    )

    save_bridge_outputs(
        results=results,
        points=points,
        colors=colors,
        source_indices=source_indices,
        extrinsics_w2c=extrinsics_w2c,
        poses_c2w=poses_c2w,
        intrinsics=intrinsics,
        render_width=render_width,
        render_height=render_height,
        args=args,
    )
    print(f"Bridge outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

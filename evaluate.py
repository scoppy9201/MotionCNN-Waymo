"""Evaluate a trained MotionCNN checkpoint and export a trajectory SVG."""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import MotionCNNDataset, get_model, postprocess_predictions
from utils import get_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute trajectory metrics and visualize one validation sample.")
    parser.add_argument("--data-path", required=True, help="Prerendered validation directory")
    parser.add_argument("--checkpoint", required=True, help="Trained .pth checkpoint")
    parser.add_argument("--config", required=True, help="Model YAML configuration")
    parser.add_argument("--output-dir", default="evaluation", help="Directory for metrics and SVG")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Validation sample to draw (default: first sample)")
    return parser.parse_args()


def local_to_global(points, shift, yaw):
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    return np.asarray(points) @ rotation.T + np.asarray(shift)


def _path(points, project):
    valid = np.asarray(points)
    if len(valid) == 0:
        return ""
    return " ".join(
        ("M" if i == 0 else "L") + f" {project(point)[0]:.1f} {project(point)[1]:.1f}"
        for i, point in enumerate(valid))


def write_trajectory_svg(path, sample, predictions, confidences, sample_number):
    history = np.asarray(sample["history_global"])
    history_valid = np.asarray(sample["history_valid"]).astype(bool)
    history = history[history_valid]
    future = np.asarray(sample["future_global"])
    future_valid = np.asarray(sample["future_valid"]).astype(bool)
    future = future[future_valid]
    predictions = [p[future_valid] for p in predictions]
    shift, yaw = sample["shift"], sample["yaw"]
    predictions = [local_to_global(p, shift, yaw) for p in predictions]

    roadgraph = np.asarray(sample.get("roadgraph_segments", np.empty((0, 2, 2))))
    if len(roadgraph) > 1500:
        roadgraph = roadgraph[np.linspace(0, len(roadgraph) - 1, 1500).astype(int)]
    all_points = [p for p in (history, future, *predictions) if len(p)]
    if len(roadgraph):
        all_points.append(roadgraph.reshape(-1, 2))
    points = np.concatenate(all_points, axis=0)
    minimum = points.min(axis=0) - 5.0
    maximum = points.max(axis=0) + 5.0
    span = np.maximum(maximum - minimum, 1.0)
    canvas = 900
    scale = min((canvas - 100) / span[0], (canvas - 140) / span[1])
    offset = np.array([50.0, 90.0])

    def project(point):
        x = offset[0] + (point[0] - minimum[0]) * scale
        y = canvas - 50.0 - (point[1] - minimum[1]) * scale
        return x, y

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="900" viewBox="0 0 900 900">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Arial" font-size="20" font-weight="bold">MotionCNN trajectory prediction</text>',
        f'<text x="28" y="60" font-family="Arial" font-size="13">Validation sample {sample_number} · confidence scores shown in legend</text>',
    ]
    for segment in roadgraph:
        xy = np.asarray(segment).reshape(-1, 2)
        x1, y1 = project(xy[0]); x2, y2 = project(xy[-1])
        pieces.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#dddddd" stroke-width="1"/>')
    if len(history) > 1:
        pieces.append(f'<path d="{_path(history, project)}" fill="none" stroke="#2563eb" stroke-width="4"/>')
    if len(future) > 0:
        pieces.append(f'<path d="{_path(future, project)}" fill="none" stroke="#111827" stroke-width="4"/>')
    colors = ("#dc2626", "#16a34a", "#9333ea")
    for mode, prediction in enumerate(predictions):
        if len(prediction):
            pieces.append(f'<path d="{_path(prediction, project)}" fill="none" stroke="{colors[mode % len(colors)]}" stroke-width="3" stroke-dasharray="8 5"/>')
    legend = [("History", "#2563eb"), ("Ground truth", "#111827")]
    legend.extend((f"Mode {i + 1} (p={float(confidences[i]):.3f})", colors[i % len(colors)])
                  for i in range(len(predictions)))
    for i, (label, color) in enumerate(legend):
        x = 30 + (i % 3) * 285
        y = 835 + (i // 3) * 25
        pieces.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        pieces.append(f'<text x="{x + 36}" y="{y + 5}" font-family="Arial" font-size="13">{label}</text>')
    pieces.append("</svg>")
    with open(path, "w", encoding="utf-8") as svg_file:
        svg_file.write("\n".join(pieces))


def main():
    args = parse_args()
    config = get_config(args.config)
    model_config = config["model"]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch install.")

    model = get_model(model_config).to(device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = MotionCNNDataset(args.data_path)
    if not len(dataset):
        raise ValueError(f"No prerendered agent .npz files found in {args.data_path}")
    if not 0 <= args.sample_index < len(dataset):
        raise ValueError(f"sample-index must be between 0 and {len(dataset) - 1}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    totals = {"min_ade": 0.0, "min_fde": 0.0, "top1_ade": 0.0, "top1_fde": 0.0}
    count = 0
    selected = None
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="Evaluating")):
            prediction = postprocess_predictions(
                model(batch["raster"].to(device=device, dtype=torch.float32)), model_config)
            pred_xy = prediction["xy"].cpu().numpy()
            confidence = prediction["confidences"].cpu().numpy()
            truth = batch["future_local"].numpy()
            valid = batch["future_valid"].numpy().astype(bool)
            distances = np.linalg.norm(pred_xy - truth[:, None, :, :], axis=-1)
            for item in range(len(truth)):
                mask = valid[item]
                if not np.any(mask):
                    continue
                mode_ade = distances[item][:, mask].mean(axis=1)
                last_index = np.flatnonzero(mask)[-1]
                mode_fde = distances[item, :, last_index]
                top_mode = int(np.argmax(confidence[item]))
                totals["min_ade"] += float(np.min(mode_ade))
                totals["min_fde"] += float(np.min(mode_fde))
                totals["top1_ade"] += float(mode_ade[top_mode])
                totals["top1_fde"] += float(mode_fde[top_mode])
                count += 1
                global_index = batch_index * args.batch_size + item
                if global_index == args.sample_index:
                    selected = (dataset[global_index], pred_xy[item], confidence[item])

    if count == 0:
        raise ValueError("Validation data has no samples with valid future positions.")
    metrics = {name: value / count for name, value in totals.items()}
    metrics.update({"num_agents": count, "checkpoint": os.path.abspath(args.checkpoint)})
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    sample, sample_predictions, sample_confidences = selected
    roadgraph_path = os.path.join(
        args.data_path, str(sample["scenario_id"]), "roadgraph_data", "segments_global.npz")
    if os.path.exists(roadgraph_path):
        with np.load(roadgraph_path) as roadgraph_file:
            sample["roadgraph_segments"] = roadgraph_file["roadgraph_segments"]
    svg_path = os.path.join(args.output_dir, "trajectory.svg")
    write_trajectory_svg(
        svg_path, sample, sample_predictions, sample_confidences, args.sample_index)
    print(json.dumps(metrics, indent=2))
    print(f"Metrics: {metrics_path}")
    print(f"Trajectory visualization: {svg_path}")


if __name__ == "__main__":
    main()

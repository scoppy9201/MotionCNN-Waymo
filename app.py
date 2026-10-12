"""Interactive MotionCNN dashboard for exploring Waymo motion predictions."""

import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml

from train import get_model, postprocess_predictions
from utils import get_config


st.set_page_config(
    page_title="MotionCNN | Quỹ đạo chuyển động",
    page_icon="🚘",
    layout="wide",
)
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; max-width: 1550px;}
      div[data-testid="stMetric"] {background:#111827; padding:14px 16px; border-radius:12px;}
      div[data-testid="stMetricLabel"] p {color:#9ca3af;}
      .small-note {color:#9ca3af; font-size:.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
MODE_COLORS = ("#fb7185", "#fbbf24", "#a78bfa", "#34d399", "#38bdf8")


@st.cache_resource(show_spinner="Đang nạp checkpoint...")
def load_model(checkpoint_path, config_path, device_name):
    config = get_config(config_path)
    model_config = config["model"]
    model = get_model(model_config).to(torch.device(device_name))
    checkpoint = torch.load(
        checkpoint_path, map_location=device_name, weights_only=True)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint does not contain model_state_dict.")
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_config, checkpoint


@st.cache_data(show_spinner=False)
def get_sample_paths(data_root):
    return sorted(str(path) for path in Path(data_root).glob("*/agent_data/*.npz"))


def load_sample(npz_path, data_root):
    with np.load(npz_path, allow_pickle=True) as source:
        sample = {key: source[key] for key in source.files}
    sample["scenario_id"] = str(np.asarray(sample["scenario_id"]).item())
    roadgraph_path = os.path.join(
        data_root, sample["scenario_id"], "roadgraph_data", "segments_global.npz")
    if os.path.exists(roadgraph_path):
        with np.load(roadgraph_path) as roadgraph_file:
            sample["roadgraph_segments"] = roadgraph_file["roadgraph_segments"]
    else:
        sample["roadgraph_segments"] = np.empty((0, 2, 2), dtype=np.float32)
    return sample


def global_to_local(points, shift, yaw):
    """Convert world positions to the target-agent-centered coordinate frame."""
    angle = -float(yaw)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return (np.asarray(points) - np.asarray(shift)) @ rotation.T


def local_to_pixels(points, scale, center):
    pixels = np.asarray(points) * float(scale) + np.asarray(center)
    return pixels[:, 0], pixels[:, 1]


def make_raster_preview(raster):
    """Colorize the prerendered 27-channel input into a readable scene image."""
    height, width = raster.shape[:2]
    image = np.full((height, width, 3), (18, 25, 39), dtype=np.uint8)

    # Roadgraph channels are stored as OpenCV BGRA values.
    road_types = raster[:, :, 1:5]
    road_rgb = road_types[:, :, 2::-1]
    road_pixels = np.any(road_types > 0, axis=2)
    image[road_pixels] = np.clip(road_rgb[road_pixels], 65, 220)
    road_mask = raster[:, :, 0] > 0
    image[road_mask] = (78, 87, 101)

    surrounding_agents = np.max(raster[:, :, 5:16], axis=2) > 0
    target_agent = np.max(raster[:, :, 16:27], axis=2) > 0
    image[surrounding_agents] = (40, 125, 190)
    image[target_agent] = (250, 165, 55)
    return image


def trajectory_figure(sample, raster, predictions, confidences,
                      selected_modes, raster_config, show_raster=True):
    height, width = raster.shape[:2]
    scale = float(raster_config["scale"])
    center = np.asarray(
        [raster_config["center_x"], raster_config["center_y"]],
        dtype=np.float32)
    figure = go.Figure()
    if show_raster:
        figure.add_trace(go.Image(z=make_raster_preview(raster), name="Đầu vào raster"))

    history_valid = np.asarray(sample["history_valid"]).astype(bool)
    history = global_to_local(sample["history_global"], sample["shift"], sample["yaw"])
    history = history[history_valid]
    hx, hy = local_to_pixels(history, scale, center)
    figure.add_trace(go.Scatter(
        x=hx, y=hy, mode="lines+markers", name="Lịch sử 1 giây",
        line=dict(color="#38bdf8", width=4), marker=dict(size=4)))

    future_valid = np.asarray(sample["future_valid"]).astype(bool)
    truth = global_to_local(sample["future_global"], sample["shift"], sample["yaw"])
    truth = truth[future_valid]
    if len(truth):
        tx, ty = local_to_pixels(truth, scale, center)
        figure.add_trace(go.Scatter(
            x=tx, y=ty, mode="lines+markers", name="Tương lai thực tế",
            line=dict(color="#f8fafc", width=4), marker=dict(size=4)))

    for mode in selected_modes:
        local_path = predictions[mode][future_valid]
        px, py = local_to_pixels(local_path, scale, center)
        prob = float(confidences[mode])
        figure.add_trace(go.Scatter(
            x=px, y=py, mode="lines+markers",
            name=f"Mode {mode + 1} · confidence {prob:.1%}",
            line=dict(color=MODE_COLORS[mode % len(MODE_COLORS)], width=3, dash="dash"),
            marker=dict(size=3)))

    figure.update_layout(
        template="plotly_dark", height=690,
        paper_bgcolor="#0b1220", plot_bgcolor="#0b1220",
        margin=dict(l=8, r=8, t=18, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(range=[0, width], visible=False, constrain="domain"),
        yaxis=dict(range=[height, 0], visible=False, scaleanchor="x", scaleratio=1),
        hovermode="closest",
    )
    return figure


def model_diagram(model_config, raster_config):
    raster_size = int(raster_config["raster_size"])
    n_modes = int(model_config["n_modes"])
    n_timestamps = int(model_config["n_timestamps"])
    output_dim = n_modes + n_modes * n_timestamps * 5
    future_seconds = n_timestamps / 10
    fig = go.Figure()
    boxes = [
        (0.2, "SCENE INPUT", f"{raster_size} × {raster_size} × 27<br>5 kênh bản đồ<br>22 kênh tác nhân", "#164e63"),
        (2.0, "ENCODER", f"{model_config['backbone']}<br>đặc trưng không gian", "#1e3a8a"),
        (3.8, "PREDICTION HEAD", f"{output_dim} giá trị<br>{n_modes} confidence logits<br>{n_modes} × {n_timestamps} × 5", "#5b21b6"),
        (5.6, "TRAJECTORY MODES", f"{n_modes} quỹ đạo<br>{n_timestamps} bước / {future_seconds:g} giây<br>(x, y, σx, σy, visibility)", "#9a3412"),
    ]
    for x, title, detail, color in boxes:
        fig.add_shape(type="rect", x0=x, x1=x + 1.45, y0=0.25, y1=1.55,
                      line=dict(color=color, width=2), fillcolor=color, opacity=.88)
        fig.add_annotation(x=x + .725, y=1.2, text=f"<b>{title}</b>",
                           showarrow=False, font=dict(size=12, color="white"))
        fig.add_annotation(x=x + .725, y=.72, text=detail, showarrow=False,
                           font=dict(size=11, color="white"))
    for x in (1.72, 3.52, 5.32):
        fig.add_annotation(x=x, y=.9, ax=x - .18, ay=.9, text="", showarrow=True,
                           arrowhead=3, arrowsize=1.2, arrowwidth=2,
                           arrowcolor="#94a3b8")
    fig.update_layout(
        template="plotly_dark", height=260, margin=dict(l=5, r=5, t=15, b=5),
        xaxis=dict(visible=False, range=[0, 7.3]),
        yaxis=dict(visible=False, range=[0, 1.8]),
        paper_bgcolor="#0b1220", plot_bgcolor="#0b1220",
    )
    return fig


def render_model_explainer(model_config, raster_config, parameter_count=None):
    n_modes = int(model_config["n_modes"])
    n_timestamps = int(model_config["n_timestamps"])
    raster_size = int(raster_config["raster_size"])
    future_seconds = n_timestamps / 10
    st.subheader("Đường đi từ cảnh giao thông đến quỹ đạo")
    st.plotly_chart(
        model_diagram(model_config, raster_config), width="stretch")
    st.markdown(f"""
    **1 · Raster hóa cảnh.** Tiền xử lý chuyển bản đồ và lịch sử các tác nhân thành ảnh {raster_size}×{raster_size},
    gồm 5 kênh bản đồ và 22 kênh vị trí tác nhân trong 11 bước lịch sử.

    **2 · Trích xuất đặc trưng.** Backbone CNN đọc raster và tạo biểu diễn không gian cho cảnh.

    **3 · Nhiều giả thuyết tương lai.** Đầu ra tạo {n_modes} mode. Mỗi mode dự đoán {n_timestamps} điểm tương lai
    ({future_seconds:g} giây ở 10 Hz), độ bất định theo hai trục và visibility. Confidence thể hiện trọng số của
    từng mode sau softmax.

    **4 · Học bằng mixture likelihood.** Loss NLLGaussian2d tối ưu xác suất của quỹ đạo thật
    dưới hỗn hợp {n_modes} phân phối Gaussian; mask bỏ qua các mốc tương lai không hợp lệ.
    """)
    a, b = st.columns(2)
    parameter_label = (
        f"{parameter_count:,}" if parameter_count is not None
        else "Chưa có checkpoint")
    a.metric("Tham số mô hình", parameter_label)
    b.metric("Backbone", model_config["backbone"])
    st.caption(
        "Mode là các phương án tương lai song song, ví dụ đi thẳng hoặc rẽ. "
        "Mô hình chưa gán nhãn ngữ nghĩa cụ thể cho từng mode."
    )


st.title("MotionCNN")
st.caption("Trực quan hóa dự đoán chuyển động trên Waymo Open Motion Dataset")

with st.sidebar:
    st.header("Nguồn dữ liệu")
    data_root = st.text_input("Validation đã prerender", "data/processed/validation")
    checkpoint_root = st.text_input("Thư mục checkpoint", "checkpoints/basic")
    config_path = st.text_input("Cấu hình mô hình", "configs/basic.yaml")
    device_options = ["auto", "cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    selected_device = st.selectbox("Thiết bị suy luận", device_options, index=0)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if selected_device == "auto" else selected_device
    st.caption(f"Thiết bị: **{device.upper()}**")
    if st.button("Làm mới danh sách mẫu", width="stretch"):
        get_sample_paths.clear()
        st.rerun()
    st.divider()
    st.caption("Quỹ đạo được vẽ trong hệ tọa độ cục bộ, chồng lên raster đầu vào.")

try:
    config = get_config(config_path)
except (OSError, ValueError, yaml.YAMLError) as error:
    st.error(f"Không đọc được cấu hình mô hình {config_path}: {error}")
    st.stop()
raster_config = config.get("prerender", {})
raster_config.setdefault("raster_size", 512)
raster_config.setdefault("scale", 3)
raster_config.setdefault("center_x", raster_config["raster_size"] / 2)
raster_config.setdefault("center_y", raster_config["raster_size"] / 2)

tab_scene, tab_model, tab_help = st.tabs(
    ["🔭 Khám phá quỹ đạo", "🧠 Cách mô hình hoạt động", "ℹ️ Hướng dẫn"])

checkpoint_files = sorted(
    glob.glob(os.path.join(checkpoint_root, "*.pth")),
    key=os.path.getmtime,
    reverse=True,
)

with tab_model:
    if checkpoint_files:
        chosen_for_info = st.selectbox(
            "Checkpoint dùng để xem thông tin", checkpoint_files,
            format_func=lambda path: os.path.basename(path), key="info_checkpoint")
        model, model_config, checkpoint_data = load_model(
            chosen_for_info, config_path, device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        render_model_explainer(model_config, raster_config, parameter_count)
        with st.expander("Thông tin lần train"):
            st.write(f"Epoch đã hoàn thành: {checkpoint_data.get('epochs_processed', 'không có thông tin')}")
            st.write(f"Số batch đã xử lý: {checkpoint_data.get('processed_batches', 'không có thông tin')}")
            st.code(chosen_for_info)
    else:
        render_model_explainer(config["model"], raster_config)
        st.info("Khi checkpoint xuất hiện, dashboard sẽ nạp số tham số và thông tin train chính xác.")

with tab_help:
    st.subheader("Cách đọc màn hình")
    st.markdown("""
    - **Lịch sử** màu xanh: vị trí hợp lệ trong khoảng 1 giây trước thời điểm hiện tại.
    - **Tương lai thực tế** màu trắng: quỹ đạo ground truth trong tối đa 8 giây.
    - **Các mode** nét đứt: giả thuyết do mạng dự đoán; confidence là trọng số tương đối.
    - **ADE/FDE** tính bằng mét trên các bước tương lai hợp lệ của mẫu đang chọn.
    - **minADE/minFDE** chọn mode tốt nhất theo từng metric; **Top confidence** dùng mode có
      confidence cao nhất. Đây là metric chẩn đoán trên split đang có, chưa phải evaluator chính thức Waymo.
    """)
    st.info("Dashboard phục vụ quan sát nghiên cứu, không phải hệ thống điều khiển xe thực tế.")

with tab_scene:
    if not checkpoint_files:
        st.info("Đang chờ checkpoint. Để train hoàn tất; sau đó chọn file `.pth` ở thanh bên.")
    elif not os.path.isdir(data_root):
        st.error(f"Không tìm thấy thư mục dữ liệu: {data_root}")
    else:
        sample_files = get_sample_paths(data_root)
        if not sample_files:
            st.error(f"Chưa có file agent `.npz` trong {data_root}")
        else:
            scenario_options = sorted({Path(path).parent.parent.name for path in sample_files})
            selected_scenario = st.selectbox(
                "Lọc theo cảnh", ["Tất cả cảnh", *scenario_options])
            if selected_scenario == "Tất cả cảnh":
                visible_paths = sample_files
            else:
                visible_paths = [p for p in sample_files
                                 if Path(p).parent.parent.name == selected_scenario]
            labels = [f"agent {Path(path).stem}" for path in visible_paths]
            available_modes = list(range(int(config["model"]["n_modes"])))
            col_select, col_modes, col_bg = st.columns([3, 2, 2])
            with col_select:
                sample_index = st.selectbox(
                    "Chọn tác nhân", range(len(visible_paths)),
                    format_func=lambda index: labels[index])
            with col_modes:
                selected_modes = st.multiselect(
                    "Các mode muốn xem", available_modes,
                    default=available_modes,
                    format_func=lambda mode: f"Mode {mode + 1}")
            with col_bg:
                show_raster = st.checkbox("Hiện raster đầu vào", value=True)

            try:
                active_checkpoint = st.selectbox(
                    "Checkpoint", checkpoint_files,
                    format_func=lambda path: os.path.basename(path), key="scene_checkpoint")
                model, model_config, checkpoint_data = load_model(
                    active_checkpoint, config_path, device)
                sample_path = visible_paths[sample_index]
                sample = load_sample(sample_path, data_root)
                raster = np.asarray(sample["raster"])
                if raster.ndim != 3 or raster.shape[2] != 27:
                    raise ValueError(
                        f"Raster phải có 27 kênh, nhận được shape {raster.shape}.")
                model_input = torch.from_numpy(
                    raster.astype(np.float32).transpose(2, 0, 1) / 255.0
                ).unsqueeze(0).to(device)
                if device == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model(model_input)
                    prediction = postprocess_predictions(output, model_config)
                if device == "cuda":
                    torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - started) * 1000
                predicted_paths = prediction["xy"][0].cpu().numpy()
                confidences = torch.softmax(
                    prediction["confidences"], dim=-1)[0].cpu().numpy()

                future_valid = np.asarray(sample["future_valid"]).astype(bool)
                ground_truth = np.asarray(sample["future_local"])
                if np.any(future_valid):
                    distances = np.linalg.norm(
                        predicted_paths - ground_truth[None, :, :], axis=-1)
                    ade_by_mode = distances[:, future_valid].mean(axis=1)
                    last_index = np.flatnonzero(future_valid)[-1]
                    fde_by_mode = distances[:, last_index]
                    top_mode = int(np.argmax(confidences))
                    metric_cols = st.columns(5)
                    metric_cols[0].metric("minADE", f"{ade_by_mode.min():.2f} m")
                    metric_cols[1].metric("minFDE", f"{fde_by_mode.min():.2f} m")
                    metric_cols[2].metric("Top confidence ADE", f"{ade_by_mode[top_mode]:.2f} m")
                    metric_cols[3].metric("Top confidence FDE", f"{fde_by_mode[top_mode]:.2f} m")
                    metric_cols[4].metric("Suy luận", f"{inference_ms:.1f} ms")
                    st.caption(
                        f"Cảnh `{sample['scenario_id']}` · agent `{sample['agent_id']}` · "
                        f"confidence mode: {top_mode + 1} ({confidences[top_mode]:.1%}) · "
                        f"tốc độ hiện tại: {float(sample['current_speed']):.1f} m/s"
                    )
                else:
                    st.warning("Mẫu này không có mốc tương lai hợp lệ để tính ADE/FDE.")

                figure = trajectory_figure(
                    sample, raster, predicted_paths, confidences,
                    selected_modes, raster_config, show_raster=show_raster)
                st.plotly_chart(figure, width="stretch")
                confidence_fig = go.Figure(go.Bar(
                    x=confidences, y=[f"Mode {i + 1}" for i in range(len(confidences))],
                    orientation="h",
                    marker_color=[MODE_COLORS[i % len(MODE_COLORS)]
                                  for i in range(len(confidences))]))
                confidence_fig.update_layout(
                    title="Confidence của các giả thuyết", template="plotly_dark",
                    height=230, xaxis=dict(range=[0, 1], title="Confidence"),
                    margin=dict(l=10, r=10, t=45, b=10),
                    paper_bgcolor="#0b1220", plot_bgcolor="#0b1220")
                st.plotly_chart(confidence_fig, width="stretch")

                predicted_global = [
                    (path @ np.asarray([[np.cos(float(sample['yaw'])), -np.sin(float(sample['yaw']))],
                                        [np.sin(float(sample['yaw'])), np.cos(float(sample['yaw']))]]).T
                     + np.asarray(sample["shift"])).tolist()
                    for path in predicted_paths]
                export = {
                    "scenario_id": sample["scenario_id"],
                    "agent_id": int(sample["agent_id"]),
                    "confidence": confidences.tolist(),
                    "predicted_global_xy": predicted_global,
                }
                st.download_button(
                    "Tải dự đoán mẫu dạng JSON", json.dumps(export, indent=2),
                    file_name=f"prediction_{sample['scenario_id']}_{sample['agent_id']}.json",
                    mime="application/json")
            except Exception as error:
                st.error(f"Không thể tạo dự đoán cho mẫu này: {error}")
                with st.expander("Chi tiết kỹ thuật"):
                    st.exception(error)

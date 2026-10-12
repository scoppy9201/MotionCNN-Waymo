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
    page_icon=":material/directions_car:",
    layout="wide",
)
st.markdown(
    """
    <style>
      :root {--ink:#152238; --muted:#52627a; --line:#d7e0ec; --blue:#1769d2; --canvas:#f5f8fc;}
      .stApp {background:var(--canvas); color:var(--ink);}
      .block-container {padding-top:1.35rem; padding-bottom:3rem; max-width:1600px;}
      h1, h2, h3 {color:var(--ink) !important; letter-spacing:-.025em;}
      p, label, [data-testid="stCaptionContainer"] {color:var(--muted);}
      [data-testid="stSidebar"] {background:#edf2f8; border-right:1px solid var(--line);}
      [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {color:var(--ink) !important;}
      [data-testid="stMetric"] {background:#fff; border:1px solid var(--line); border-top:4px solid var(--blue); padding:15px 17px; border-radius:13px; box-shadow:0 3px 12px #1b355010;}
      [data-testid="stMetricLabel"] p {color:#52627a !important; font-size:.9rem; font-weight:650;}
      [data-testid="stMetricValue"] {color:#10213b !important; font-weight:750;}
      [data-baseweb="select"] > div {background:#fff; border-color:#cbd5e1; color:#152238;}
      [data-testid="stTextInput"] input {background:#fff; color:#152238;}
      [data-testid="stTabs"] button[role="tab"] {color:#42536c; font-weight:650;}
      [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {color:var(--blue);}
      [data-testid="stAlert"] {border-radius:12px;}
      [data-testid="stPlotlyChart"] {background:#fff; border:1px solid var(--line); border-radius:14px; padding:8px;}
      .motion-hero {background:linear-gradient(115deg,#e9f3ff 0%,#fff 72%); border:1px solid #c9def8; border-left:5px solid #1769d2; border-radius:14px; padding:16px 20px; margin:.5rem 0 1.25rem;}
      .motion-hero-title {color:#17365f; font-size:1.08rem; font-weight:750; margin-bottom:4px;}
      .motion-hero-copy {color:#42536c; font-size:.96rem; margin:0;}
      .section-intro {color:#52627a; margin-top:-.45rem; margin-bottom:1rem;}
      .demo-guide {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:.75rem 0 1rem;}
      .demo-step {background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px 14px; box-shadow:0 2px 8px #1b355008;}
      .demo-step strong {display:block; color:#17365f; margin-bottom:3px;}
      .demo-step span {color:#52627a; font-size:.9rem;}
      .result-callout {background:#eaf7f0; border:1px solid #b9e2ca; border-left:5px solid #16824b; border-radius:12px; padding:13px 16px; margin:.5rem 0 1rem; color:#174c32;}
      @media (max-width: 760px) {.demo-guide {grid-template-columns:1fr;}}
    </style>
    """,
    unsafe_allow_html=True,
)
MODE_COLORS = ("#c62845", "#a96800", "#6f42c1", "#087f5b", "#1464b4")


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
    image = np.full((height, width, 3), (245, 248, 252), dtype=np.uint8)

    # Roadgraph channels are stored as OpenCV BGRA values.
    road_types = raster[:, :, 1:5]
    road_rgb = road_types[:, :, 2::-1]
    road_pixels = np.any(road_types > 0, axis=2)
    image[road_pixels] = np.clip(road_rgb[road_pixels], 65, 220)
    road_mask = raster[:, :, 0] > 0
    image[road_mask] = (103, 116, 133)

    surrounding_agents = np.max(raster[:, :, 5:16], axis=2) > 0
    target_agent = np.max(raster[:, :, 16:27], axis=2) > 0
    image[surrounding_agents] = (33, 103, 169)
    image[target_agent] = (222, 105, 22)
    return image


def trajectory_figure(sample, raster, predictions, confidences,
                      selected_modes, raster_config, show_raster=True,
                      view_range_m=100):
    """Draw history, observed future, and selected model hypotheses on the scene."""
    scale = float(raster_config["scale"])
    center = np.asarray(
        [raster_config["center_x"], raster_config["center_y"]], dtype=np.float32)
    figure = go.Figure()
    if show_raster:
        figure.add_trace(go.Image(z=make_raster_preview(raster), name="Bản đồ cảnh"))

    history_valid = np.asarray(sample["history_valid"]).astype(bool)
    history = global_to_local(
        sample["history_global"], sample["shift"], sample["yaw"])[history_valid]
    hx, hy = local_to_pixels(history, scale, center)
    history_indices = np.flatnonzero(history_valid)
    history_hover = np.column_stack(
        ((history_indices - (len(history_valid) - 1)) / 10.0, history))
    figure.add_trace(go.Scatter(
        x=hx, y=hy, mode="lines+markers", name="Đã đi · 1 giây trước",
        customdata=history_hover,
        hovertemplate="Thời gian: %{customdata[0]:.1f} s<br>x: %{customdata[1]:.1f} m<br>y: %{customdata[2]:.1f} m<extra>%{fullData.name}</extra>",
        line=dict(color="#1464b4", width=5),
        marker=dict(size=5, color="#1464b4", line=dict(color="#ffffff", width=1))))
    if len(history):
        current_x, current_y = local_to_pixels(history[-1:], scale, center)
        figure.add_trace(go.Scatter(
            x=current_x, y=current_y, mode="markers+text", name="Vị trí hiện tại",
            text=["HIỆN TẠI"], textposition="top center",
            textfont=dict(color="#152238", size=12),
            marker=dict(symbol="circle", size=15, color="#ffd43b",
                        line=dict(color="#152238", width=2)),
            hovertemplate="Vị trí hiện tại<extra></extra>"))

    future_valid = np.asarray(sample["future_valid"]).astype(bool)
    future_indices = np.flatnonzero(future_valid)
    future_times = (future_indices + 1) / 10.0
    truth = global_to_local(
        sample["future_global"], sample["shift"], sample["yaw"])[future_valid]
    if len(truth):
        tx, ty = local_to_pixels(truth, scale, center)
        truth_hover = np.column_stack((future_times, truth))
        figure.add_trace(go.Scatter(
            x=tx, y=ty, mode="lines+markers", name="Đã xảy ra · quỹ đạo thật",
            customdata=truth_hover,
            hovertemplate="Thời gian: +%{customdata[0]:.1f} s<br>x: %{customdata[1]:.1f} m<br>y: %{customdata[2]:.1f} m<extra>%{fullData.name}</extra>",
            line=dict(color="#152238", width=5),
            marker=dict(size=5, color="#152238", line=dict(color="#ffffff", width=1))))

    for mode in selected_modes:
        local_path = predictions[mode][future_valid]
        px, py = local_to_pixels(local_path, scale, center)
        probability = float(confidences[mode])
        mode_hover = np.column_stack((future_times, local_path))
        figure.add_trace(go.Scatter(
            x=px, y=py, mode="lines+markers",
            name=f"Dự đoán {mode + 1} · độ tin cậy {probability:.0%}",
            customdata=mode_hover,
            hovertemplate="Dự đoán sau %{customdata[0]:.1f} s<br>x: %{customdata[1]:.1f} m<br>y: %{customdata[2]:.1f} m<extra>%{fullData.name}</extra>",
            line=dict(color=MODE_COLORS[mode % len(MODE_COLORS)], width=4, dash="dash"),
            marker=dict(size=4)))

    half_view = float(view_range_m) * scale / 2
    figure.update_layout(
        template="plotly_white", height=690,
        font=dict(family="Arial, sans-serif", size=13, color="#152238"),
        paper_bgcolor="#ffffff", plot_bgcolor="#f3f6fa",
        margin=dict(l=8, r=8, t=56, b=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            font=dict(size=12, color="#152238"),
            bgcolor="rgba(255,255,255,.96)", bordercolor="#cbd5e1", borderwidth=1),
        xaxis=dict(
            range=[center[0] - half_view, center[0] + half_view],
            visible=False, constrain="domain"),
        yaxis=dict(
            range=[center[1] + half_view, center[1] - half_view],
            visible=False, scaleanchor="x", scaleratio=1),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor="#1769d2",
            font=dict(color="#152238", size=13)),
    )
    return figure

def model_diagram(model_config, raster_config):
    raster_size = int(raster_config["raster_size"])
    n_modes = int(model_config["n_modes"])
    n_timestamps = int(model_config["n_timestamps"])
    future_seconds = n_timestamps / 10
    fig = go.Figure()
    boxes = [
        (0.2, "CẢNH GIAO THÔNG", f"Ảnh {raster_size} × {raster_size}<br>đường + xe xung quanh", "#164e63"),
        (2.0, "MẠNG CNN", f"{model_config['backbone']}<br>nhận biết bố cục cảnh", "#1e3a8a"),
        (3.8, "ĐẦU DỰ ĐOÁN", f"{n_modes} khả năng<br>mỗi khả năng có xác suất", "#5b21b6"),
        (5.6, "ĐƯỜNG ĐI TƯƠNG LAI", f"{n_modes} quỹ đạo<br>{future_seconds:g} giây tiếp theo", "#9a3412"),
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
    st.header("Đường đi từ cảnh giao thông đến quỹ đạo", icon=":material/neurology:")
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


st.title("MotionCNN", icon=":material/directions_car:")
st.markdown(
    """
    <div class="motion-hero">
      <div class="motion-hero-title">Xem xe đã đi đâu và mô hình dự đoán xe sẽ đi về đâu</div>
      <p class="motion-hero-copy">Chọn một cảnh và một xe; dự đoán tự chạy ngay. Bản đồ cho thấy lịch sử, ba hướng dự đoán và đường thật để đối chiếu.
      Đây là demo suy luận với checkpoint có sẵn; mở giao diện không bắt đầu train lại.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Demo chuyển động", icon=":material/directions_car:")
    st.caption("Dùng mô hình đã lưu · chỉ dự đoán, không huấn luyện lại")
    with st.expander("Cấu hình nâng cao", expanded=False):
        data_root = st.text_input("Thư mục dữ liệu validation", "data/processed/validation")
        checkpoint_root = st.text_input("Thư mục checkpoint", "checkpoints/basic")
        config_path = st.text_input("File cấu hình", "configs/basic.yaml")
        device_options = ["auto", "cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        selected_device = st.selectbox("Thiết bị dự đoán", device_options, index=0)
        if st.button("Làm mới danh sách mẫu", icon=":material/refresh:", width="stretch"):
            get_sample_paths.clear()
            st.rerun()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if selected_device == "auto" else selected_device
    st.caption(f"Thiết bị: **{device.upper()}**")
    st.divider()
    st.caption("Chọn xe trên màn hình chính để xem dự đoán ngay.")

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

tab_scene, tab_eval, tab_model, tab_help = st.tabs(
    ["Khám phá quỹ đạo", "Kết quả validation", "Mô hình hoạt động", "Hướng dẫn"])

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
        with st.expander("Thông tin lần huấn luyện"):
            st.write(f"Epoch đã hoàn thành: {checkpoint_data.get('epochs_processed', 'không có thông tin')}")
            st.write(f"Số batch đã xử lý: {checkpoint_data.get('processed_batches', 'không có thông tin')}")
            st.code(chosen_for_info)
    else:
        render_model_explainer(config["model"], raster_config)
        st.info("Khi checkpoint xuất hiện, dashboard sẽ nạp số tham số và thông tin train chính xác.")

with tab_eval:
    evaluation_files = sorted(
        glob.glob(os.path.join("evaluation", "**", "metrics.json"), recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    if not evaluation_files:
        st.info("Chưa có báo cáo validation. Sau khi chạy evaluate.py, metrics sẽ hiện ở đây.")
    else:
        selected_metrics_file = st.selectbox(
            "Bản đánh giá", evaluation_files,
            format_func=lambda path: Path(path).parent.name,
        )
        try:
            with open(selected_metrics_file, "r", encoding="utf-8") as metrics_stream:
                evaluation_metrics = json.load(metrics_stream)
            eval_cols = st.columns(4)
            eval_cols[0].metric(
                "Sai lệch TB · gần nhất",
                f"{evaluation_metrics['min_ade']:.2f} m",
                help="minADE: trung bình khoảng cách của đường dự đoán gần thực tế nhất trong 3 hướng.")
            eval_cols[1].metric(
                "Sai lệch cuối · gần nhất",
                f"{evaluation_metrics['min_fde']:.2f} m",
                help="minFDE: sai lệch tại điểm cuối của đường dự đoán gần thực tế nhất trong 3 hướng.")
            eval_cols[2].metric(
                "Sai lệch TB · mô hình chọn",
                f"{evaluation_metrics['top1_ade']:.2f} m",
                help="top1 ADE: sai lệch trung bình của đường có xác suất cao nhất.")
            eval_cols[3].metric(
                "Sai lệch cuối · mô hình chọn",
                f"{evaluation_metrics['top1_fde']:.2f} m",
                help="top1 FDE: sai lệch tại điểm cuối của đường có xác suất cao nhất.")

            comparison = go.Figure()
            comparison.add_trace(go.Bar(
                name="Gần thực tế nhất trong 3 hướng",
                x=["Sai lệch trung bình (ADE)", "Sai lệch tại điểm cuối (FDE)"],
                y=[evaluation_metrics["min_ade"], evaluation_metrics["min_fde"]],
                marker_color="#1769d2",
            ))
            comparison.add_trace(go.Bar(
                name="Hướng mô hình ưu tiên",
                x=["Sai lệch trung bình (ADE)", "Sai lệch tại điểm cuối (FDE)"],
                y=[evaluation_metrics["top1_ade"], evaluation_metrics["top1_fde"]],
                marker_color="#d97706",
            ))
            comparison.update_layout(
                title="Mô hình chọn hướng có xác suất cao so với hướng tốt nhất",
                barmode="group", template="plotly_white", height=390,
                yaxis_title="Sai lệch (m)",
                font=dict(family="Arial, sans-serif", size=13, color="#152238"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(l=10, r=10, t=70, b=10),
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            )
            st.plotly_chart(comparison, width="stretch")
            st.caption(
                f"Đánh giá trên {evaluation_metrics['num_agents']:,} mẫu có quỹ đạo tương lai hợp lệ · "
                f"checkpoint: {Path(evaluation_metrics['checkpoint']).name}. "
                "“Gần thực tế nhất” là chỉ số tham khảo kiểu oracle vì chọn đường sau khi đã biết đáp án; "
                "“mô hình ưu tiên” phản ánh đường được mạng chọn theo xác suất."
            )
            st.info(
                "Đây là kết quả trên phần dữ liệu Waymo đã tải và prerender cho demo, "
                "không phải điểm benchmark chính thức trên toàn bộ Waymo."
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            st.error(f"Không đọc được báo cáo validation: {error}")

with tab_help:
    st.header("Cách đọc màn hình", icon=":material/help_outline:")
    st.info(
        "Để trình bày: mở mục Khám phá quỹ đạo, chọn tình huống rồi chọn xe/người. "
        "Dự đoán cập nhật tự động bằng checkpoint có sẵn; giao diện không chạy lại quá trình train."
    )
    st.markdown("""
    - **Lịch sử** màu xanh: vị trí hợp lệ trong khoảng 1 giây trước thời điểm hiện tại.
    - **Quỹ đạo thật** màu xanh đậm: đường xe thực sự đi trong tối đa 8 giây tiếp theo.
    - **Các dự đoán** nét đứt nhiều màu: những hướng khác nhau mạng đưa ra; độ tin cậy cho biết mạng ưu tiên hướng nào.
    - **ADE** là độ lệch trung bình dọc quãng đường; **FDE** là độ lệch tại điểm cuối. Cả hai tính bằng mét.
    - “Tốt nhất” chọn đường dự đoán gần quỹ đạo thật nhất; “mô hình ưu tiên” chọn đường có độ tin cậy cao nhất.
    - Các con số trên đang đo một mẫu để tham khảo, chưa phải kết quả đánh giá chính thức của Waymo.
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
            st.markdown("#### :material/route: Bắt đầu khám phá")
            st.markdown("""
            <div class="demo-guide">
              <div class="demo-step"><strong>1 · Chọn cảnh</strong><span>Chọn một tình huống giao thông từ dữ liệu validation.</span></div>
              <div class="demo-step"><strong>2 · Chọn xe</strong><span>Chọn đối tượng muốn theo dõi trong cảnh đó.</span></div>
              <div class="demo-step"><strong>3 · Đọc đường đi</strong><span>So sánh lịch sử, các dự đoán và quỹ đạo thật.</span></div>
            </div>
            """, unsafe_allow_html=True)
            scenario_options = sorted({Path(path).parent.parent.name for path in sample_files})
            scenario_labels = ["Tất cả cảnh", *[
                f"Tình huống {index + 1}" for index in range(len(scenario_options))]]
            selected_scenario_label = st.selectbox(
                "1 · Chọn tình huống", scenario_labels,
                index=1 if scenario_options else 0)
            st.caption(
                f"Bộ validation có {len(scenario_options):,} tình huống và "
                f"{len(sample_files):,} mẫu xe/người.")
            if selected_scenario_label == "Tất cả cảnh":
                visible_paths = sample_files
            else:
                selected_scenario = scenario_options[
                    scenario_labels.index(selected_scenario_label) - 1]
                visible_paths = [p for p in sample_files
                                 if Path(p).parent.parent.name == selected_scenario]
            labels = [
                f"Xe/người {index + 1} · mã {Path(path).stem}"
                for index, path in enumerate(visible_paths)]
            available_modes = list(range(int(config["model"]["n_modes"])))
            col_select, col_modes, col_bg, col_zoom = st.columns([3, 2, 1.2, 1.5])
            with col_select:
                sample_index = st.selectbox(
                    "2 · Chọn đối tượng cần theo dõi", range(len(visible_paths)),
                    format_func=lambda index: labels[index])
            with col_modes:
                selected_modes = st.multiselect(
                    "3 · Các hướng dự đoán", available_modes,
                    default=available_modes,
                    format_func=lambda mode: f"Dự đoán {mode + 1}")
            with col_bg:
                show_raster = st.checkbox("Hiện bản đồ nền", value=True)
            with col_zoom:
                view_range_m = st.select_slider(
                    "Phạm vi bản đồ", options=[60, 80, 100, 120, 150],
                    value=100, format_func=lambda value: f"{value} m")

            try:
                active_checkpoint = st.selectbox(
                    "Mô hình đang dùng để dự đoán", checkpoint_files,
                    format_func=lambda path: os.path.basename(path), key="scene_checkpoint")
                model, model_config, checkpoint_data = load_model(
                    active_checkpoint, config_path, device)
                st.info(
                    f"Đang dùng `{os.path.basename(active_checkpoint)}` · "
                    f"đã huấn luyện {checkpoint_data.get('epochs_processed', '?')} epoch, "
                    f"{checkpoint_data.get('processed_batches', '?')} batch. "
                    "Mở giao diện chỉ chạy dự đoán, không train lại mô hình.",
                    icon=":material/model_training:",
                )
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
                    metric_cols[0].metric(
                        "Độ lệch trung bình tốt nhất",
                        f"{ade_by_mode.min():.2f} m",
                        help="minADE: trung bình khoảng cách giữa một hướng dự đoán tốt nhất và quỹ đạo thật. Thấp hơn là tốt hơn.",
                        icon=":material/straighten:")
                    metric_cols[1].metric(
                        "Độ lệch điểm cuối tốt nhất",
                        f"{fde_by_mode.min():.2f} m",
                        help="minFDE: khoảng cách tại thời điểm cuối giữa hướng dự đoán tốt nhất và quỹ đạo thật. Thấp hơn là tốt hơn.",
                        icon=":material/flag:")
                    metric_cols[2].metric(
                        "Độ lệch TB · hướng mô hình ưu tiên",
                        f"{ade_by_mode[top_mode]:.2f} m",
                        help="Độ lệch trung bình của hướng có độ tin cậy cao nhất. Đây là hướng mô hình ưu tiên.",
                        icon=":material/route:")
                    metric_cols[3].metric(
                        "Độ lệch cuối · hướng mô hình ưu tiên",
                        f"{fde_by_mode[top_mode]:.2f} m",
                        help="Độ lệch tại điểm cuối của hướng có độ tin cậy cao nhất. Đây là hướng mô hình ưu tiên.",
                        icon=":material/place:")
                    metric_cols[4].metric(
                        "Thời gian dự đoán", f"{inference_ms:.1f} ms",
                        help="Thời gian chạy một mẫu trên thiết bị đang chọn; không bao gồm tải mô hình lần đầu.",
                        icon=":material/speed:")
                    st.markdown(
                        f"<div class='result-callout'><strong>Mô hình đang ưu tiên Dự đoán {top_mode + 1} "
                        f"({confidences[top_mode]:.1%}).</strong> Với mẫu này, đường được ưu tiên lệch trung bình "
                        f"{ade_by_mode[top_mode]:.2f} m so với đường thật ghi trong dữ liệu. Đường thật chỉ dùng để "
                        "đối chiếu sau khi dự đoán, không phải đầu vào của mô hình.</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption("ADE = sai lệch trung bình dọc quãng đường · FDE = sai lệch ở điểm cuối. “Tốt nhất” chọn trong các dự đoán đường gần thực tế nhất; “mô hình ưu tiên” là đường có xác suất cao nhất.")
                    st.caption(
                        f"Cảnh `{sample['scenario_id']}` · đối tượng `{sample['agent_id']}` · "
                        f"hướng ưu tiên: dự đoán {top_mode + 1} (độ tin cậy {confidences[top_mode]:.1%}) · "
                        f"tốc độ hiện tại: {float(sample['current_speed']):.1f} m/s"
                    )
                else:
                    st.warning("Mẫu này không có mốc tương lai hợp lệ để tính ADE/FDE.")

                st.markdown("#### :material/alt_route: So sánh đường đi")
                st.caption("Nền: đường màu xám, xe xung quanh màu xanh, xe đang theo dõi màu cam. Đường xanh liền là lịch sử; đường đen là thực tế để so sánh; đường nét đứt nhiều màu là dự đoán. Chấm vàng đánh dấu vị trí hiện tại. Rê chuột lên đường để xem thời gian và tọa độ.")
                figure = trajectory_figure(
                    sample, raster, predicted_paths, confidences,
                    selected_modes, raster_config, show_raster=show_raster,
                    view_range_m=view_range_m)
                st.plotly_chart(figure, width="stretch")
                st.markdown("#### :material/monitoring: Mức độ tin cậy của các dự đoán")
                st.caption("Thanh dài hơn nghĩa là mô hình ưu tiên phương án đó hơn. Các mức tin cậy cộng lại thành 100%.")
                confidence_fig = go.Figure(go.Bar(
                    x=confidences, y=[f"Dự đoán {i + 1}" for i in range(len(confidences))],
                    orientation="h",
                    text=[f"{value:.1%}" for value in confidences],
                    textposition="outside",
                    textfont=dict(color="#152238", size=12),
                    marker_color=[MODE_COLORS[i % len(MODE_COLORS)]
                                  for i in range(len(confidences))]))
                confidence_fig.update_layout(
                    title="Mô hình đang nghiêng về hướng nào?", template="plotly_white",
                    font=dict(family="Arial, sans-serif", size=13, color="#152238"),
                    height=260, xaxis=dict(range=[0, 1.15], title="Mức độ tin cậy", tickformat=".0%"),
                    yaxis=dict(tickfont=dict(color="#152238", size=12)),
                    margin=dict(l=10, r=42, t=45, b=10),
                    hoverlabel=dict(bgcolor="#ffffff", bordercolor="#1769d2", font=dict(color="#152238")),
                    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
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
                    mime="application/json", icon=":material/download:")
            except Exception as error:
                st.error(f"Không thể tạo dự đoán cho mẫu này: {error}")
                with st.expander("Chi tiết kỹ thuật"):
                    st.exception(error)

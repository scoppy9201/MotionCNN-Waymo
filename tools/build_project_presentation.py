"""Build the current project overview deck as a 16:9 PDF."""

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "img" / "waymo_motion_prediction_2021_3rd_place_solution_presentation.pdf"
PAGE_W, PAGE_H = 960, 540
BG = HexColor("#0B1220")
PANEL = HexColor("#111C2E")
PANEL_2 = HexColor("#162338")
WHITE = HexColor("#F8FAFC")
MUTED = HexColor("#A8B4C7")
CYAN = HexColor("#38BDF8")
AMBER = HexColor("#FBBF24")
ROSE = HexColor("#FB7185")
PURPLE = HexColor("#A78BFA")
GREEN = HexColor("#4ADE80")

FONT_CANDIDATES = {
    "Arial": (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ),
    "Arial-Bold": (
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ),
    "Arial-Italic": (
        Path(r"C:\Windows\Fonts\ariali.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Italic.ttf"),
    ),
}


def register_fonts():
    for font_name, candidates in FONT_CANDIDATES.items():
        font_path = next((path for path in candidates if path.is_file()), None)
        if font_path is None:
            raise FileNotFoundError(
                f"No Unicode font found for {font_name}; install Arial or DejaVu Sans.")
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))


register_fonts()


def ptext(c, text, x, top, width, size=14, color=WHITE, leading=None,
          font="Arial", max_height=400):
    style = ParagraphStyle(
        "deck", fontName=font, fontSize=size, leading=leading or size * 1.38,
        textColor=color, alignment=TA_LEFT, spaceAfter=0, spaceBefore=0)
    paragraph = Paragraph(text, style)
    _, height = paragraph.wrap(width, max_height)
    paragraph.drawOn(c, x, PAGE_H - top - height)
    return height


def text(c, value, x, top, size=14, color=WHITE, font="Arial"):
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawString(x, PAGE_H - top - size, value)


def line(c, x1, y1, x2, y2, color=MUTED, width=1, dash=None):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    if dash:
        c.setDash(dash)
    else:
        c.setDash()
    c.line(x1, PAGE_H - y1, x2, PAGE_H - y2)
    c.setDash()


def rect(c, x, top, width, height, fill=PANEL, stroke=None, radius=14):
    c.setFillColor(fill)
    c.setStrokeColor(stroke or fill)
    c.setLineWidth(1)
    c.roundRect(x, PAGE_H - top - height, width, height, radius, fill=1, stroke=bool(stroke))


def header(c, number, title, subtitle=None, section="PROJECT IMPLEMENTATION"):
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    text(c, section.upper(), 54, 26, 9, CYAN, "Arial-Bold")
    text(c, title, 54, 54, 25, WHITE, "Arial-Bold")
    if subtitle:
        ptext(c, subtitle, 54, 91, 820, 11, MUTED, leading=15)
    line(c, 54, 506, 906, 506, HexColor("#29374B"), .8)
    text(c, "MotionCNN Waymo | Bản triển khai dự án", 54, 515, 8, MUTED)
    text(c, f"{number:02d} / 09", 858, 515, 8, MUTED)


def card(c, x, top, width, height, title, body, accent=CYAN,
         title_size=14, body_size=11):
    rect(c, x, top, width, height)
    rect(c, x, top, 5, height, accent, radius=2)
    ptext(c, f"<b>{title}</b>", x + 18, top + 15, width - 35,
          title_size, WHITE, leading=title_size * 1.3)
    ptext(c, body, x + 18, top + 48, width - 35,
          body_size, MUTED, leading=body_size * 1.45)


def arrow(c, x1, y, x2, color=MUTED):
    line(c, x1, y, x2 - 9, y, color, 2)
    line(c, x2 - 9, y, x2 - 16, y - 5, color, 2)
    line(c, x2 - 9, y, x2 - 16, y + 5, color, 2)


def slide_cover(c):
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    text(c, "MOTIONCNN WAYMO", 65, 72, 11, CYAN, "Arial-Bold")
    ptext(c, "Bản triển khai<br/>dự đoán quỹ đạo", 65, 126, 450,
          36, WHITE, leading=45, font="Arial-Bold")
    ptext(c, "Pipeline nghiên cứu, đánh giá ADE/FDE và dashboard tương tác cho Waymo Open Motion Dataset.",
          68, 245, 405, 16, MUTED, leading=23)
    for i, (label, color) in enumerate((("Python 3.10", CYAN), ("ResNet-34", PURPLE),
                                        ("Dashboard Streamlit", AMBER))):
        rect(c, 68 + i * 160, 348, 146, 34, PANEL_2, radius=17)
        text(c, label, 82 + i * 160, 358, 10, color, "Arial-Bold")

    # Original vector illustration: road fork and three future hypotheses.
    rect(c, 526, 54, 368, 402, PANEL, radius=26)
    for lane_x in (585, 835):
        line(c, lane_x, 102, lane_x + 20, 422, HexColor("#34445C"), 2, [8, 8])
    line(c, 594, 388, 650, 320, HexColor("#64748B"), 9)
    line(c, 650, 320, 686, 250, HexColor("#64748B"), 9)
    line(c, 686, 250, 696, 120, HexColor("#64748B"), 9)
    line(c, 686, 250, 772, 165, HexColor("#64748B"), 9)
    line(c, 686, 250, 638, 155, HexColor("#64748B"), 9)
    line(c, 684, 285, 704, 165, CYAN, 4)
    line(c, 684, 285, 775, 187, ROSE, 4, [7, 5])
    line(c, 684, 285, 650, 176, AMBER, 4, [7, 5])
    rect(c, 658, 286, 52, 28, HexColor("#F97316"), radius=8)
    text(c, "3 giả thuyết tương lai", 610, 416, 12, WHITE, "Arial-Bold")
    text(c, "Minh họa vector của pipeline hiện tại", 610, 438, 9, MUTED)
    line(c, 65, 494, 895, 494, HexColor("#29374B"), .8)
    text(c, "Nội dung và kết quả trình bày ở đây thuộc bản triển khai dự án; không đại diện benchmark chính thức.",
         65, 506, 8, MUTED)
    c.showPage()


def slide_scope(c):
    header(c, 2, "Phạm vi bản triển khai",
           "Tập trung vào khả năng chạy lại pipeline và quan sát kết quả trên dữ liệu cục bộ.")
    card(c, 54, 143, 265, 202, "01 · Dữ liệu",
         "Waymo Motion v1.0.0, định dạng TFExample. Tiền xử lý tạo raster 512 × 512 × 27 và file NPZ theo agent.", CYAN)
    card(c, 347, 143, 265, 202, "02 · Mô hình",
         "ResNet-34 dự đoán 3 mode, mỗi mode gồm 80 mốc tương lai. Train hỗ trợ CPU hoặc CUDA.", PURPLE)
    card(c, 640, 143, 265, 202, "03 · Quan sát",
         "Dashboard chọn cảnh, xem lịch sử, ground truth, dự đoán, confidence và ADE/FDE theo mẫu.", AMBER)
    rect(c, 54, 375, 851, 77, HexColor("#17263A"))
    text(c, "TRẠNG THÁI PHÁT TRIỂN", 74, 393, 10, GREEN, "Arial-Bold")
    ptext(c, "Đã prerender một shard cục bộ: 2.640 mẫu agent train và 1.427 mẫu validation. Đây là quy mô thử pipeline, không phải kết quả toàn bộ dataset hay benchmark.",
          74, 416, 800, 11, WHITE, leading=15)
    c.showPage()


def slide_data(c):
    header(c, 3, "Từ TFRecord đến mẫu học",
           "Mỗi TFRecord chứa cảnh giao thông; pipeline tạo đầu vào ảnh và nhãn chuyển động cho từng agent.")
    steps = [
        ("01", "TFRecord", "Waymo Motion<br/>v1.0.0", CYAN),
        ("02", "Giải mã cảnh", "Tracks +<br/>roadgraph", PURPLE),
        ("03", "Raster hóa", "512 × 512<br/>27 channels", AMBER),
        ("04", "NPZ theo agent", "history 11<br/>future 80", GREEN),
    ]
    x_positions = (56, 278, 500, 722)
    for i, (number, title, detail, color) in enumerate(steps):
        x = x_positions[i]
        rect(c, x, 175, 180, 145, PANEL)
        text(c, number, x + 17, 193, 10, color, "Arial-Bold")
        ptext(c, f"<b>{title}</b>", x + 17, 222, 145, 15, WHITE)
        ptext(c, detail, x + 17, 258, 140, 12, MUTED, leading=17)
        if i < 3:
            arrow(c, x + 185, 247, x + 215)
    card(c, 56, 356, 395, 96, "Hệ tọa độ", "Cảnh được căn theo agent mục tiêu; vị trí tương lai được lưu cả global và local để train và hiển thị.", CYAN, 12, 10)
    card(c, 473, 356, 429, 96, "Mask hợp lệ", "Mỗi mốc tương lai có cờ valid riêng. Metric được tính trên các mốc hợp lệ của từng agent.", AMBER, 12, 10)
    c.showPage()


def slide_model(c):
    header(c, 4, "Kiến trúc mô hình hiện tại",
           "Một CNN backbone đọc raster của cảnh và tạo ba giả thuyết quỹ đạo song song.")
    blocks = [
        (55, "ĐẦU VÀO", "512 × 512 × 27\n5 map + 22 agent", CYAN),
        (270, "ENCODER", "ResNet-34\nđặc trưng không gian", PURPLE),
        (485, "HEAD", "1203 giá trị\n3 logits + 1200 outputs", AMBER),
        (700, "3 MODE", "3 × 80 × 5\nxy, covariance, visibility", ROSE),
    ]
    for x, title, detail, color in blocks:
        rect(c, x, 180, 175, 138, PANEL, stroke=color)
        text(c, title, x + 16, 197, 10, color, "Arial-Bold")
        for j, row in enumerate(detail.split("\n")):
            text(c, row, x + 16, 235 + j * 23, 12 if j == 0 else 10,
                 WHITE if j == 0 else MUTED, "Arial-Bold" if j == 0 else "Arial")
    for x in (238, 453, 668):
        arrow(c, x, 248, x + 28)
    rect(c, 55, 355, 820, 82, HexColor("#17263A"))
    ptext(c, "80 mốc ở 10 Hz tương ứng tối đa 8 giây tương lai. Mỗi mode xuất tọa độ (x, y), hai phần tử covariance đường chéo và visibility. Confidence được chuẩn hóa qua softmax.",
          76, 373, 775, 12, WHITE, leading=17)
    text(c, "Khởi tạo hiện tại: pretrained = false (train từ khởi tạo ngẫu nhiên).", 76, 414, 10, AMBER)
    c.showPage()


def slide_objective(c):
    header(c, 5, "Loss và cách đọc ADE/FDE",
           "Đánh giá quỹ đạo trong hệ tọa độ mét, chỉ xét các mốc tương lai có nhãn hợp lệ.")
    card(c, 55, 145, 400, 245, "Huấn luyện · Gaussian mixture NLL",
         "Model phát confidence cho 3 mode và covariance Gaussian đường chéo theo thời gian. NLL tối ưu likelihood của quỹ đạo thật dưới hỗn hợp các mode; residual vị trí được nhân với future_valid.", PURPLE, 14, 12)
    card(c, 485, 145, 420, 112, "ADE · Average Displacement Error",
         "Khoảng cách Euclid trung bình trên các mốc tương lai hợp lệ. Báo minADE (mode tốt nhất) và top-confidence ADE.", CYAN, 13, 11)
    card(c, 485, 278, 420, 112, "FDE · Final Displacement Error",
         "Khoảng cách tại mốc tương lai hợp lệ cuối cùng. Báo minFDE và top-confidence FDE.", AMBER, 13, 11)
    ptext(c, "Metric hiện tại dùng validation shard cục bộ để kiểm tra và phân tích. Không so sánh trực tiếp với leaderboard Waymo.",
          59, 418, 830, 11, MUTED, leading=15)
    c.showPage()


def slide_dashboard(c):
    header(c, 6, "Dashboard trực quan",
           "Giao diện Streamlit cho phép chọn cảnh và agent, bật/tắt raster, chọn các mode cần so sánh.")
    rect(c, 54, 137, 560, 315, HexColor("#0A0F19"), stroke=HexColor("#334155"), radius=16)
    # A small, hand-drawn road scene used as a deck illustration.
    for x in (155, 504):
        line(c, x, 182, x + (15 if x < 300 else -15), 407, HexColor("#475569"), 2, [7, 7])
    line(c, 167, 361, 283, 275, HexColor("#68758A"), 13)
    line(c, 283, 275, 298, 181, HexColor("#68758A"), 13)
    line(c, 283, 275, 411, 202, HexColor("#68758A"), 13)
    line(c, 283, 275, 223, 198, HexColor("#68758A"), 13)
    line(c, 282, 295, 297, 195, CYAN, 4)
    line(c, 282, 295, 402, 220, ROSE, 3, [6, 4])
    line(c, 282, 295, 230, 213, AMBER, 3, [6, 4])
    rect(c, 267, 293, 34, 21, HexColor("#F97316"), radius=5)
    text(c, "Lịch sử", 90, 421, 9, CYAN, "Arial-Bold")
    text(c, "Thực tế", 178, 421, 9, WHITE, "Arial-Bold")
    text(c, "3 mode dự đoán", 262, 421, 9, AMBER, "Arial-Bold")
    text(c, "Hình minh họa giao diện, không phải ảnh kết quả inference", 81, 438, 8, MUTED)
    card(c, 642, 137, 263, 95, "Theo dõi sai số", "minADE / minFDE và mode confidence cao nhất.", CYAN, 12, 10)
    card(c, 642, 246, 263, 95, "Tương tác", "Lọc cảnh, chọn agent, bật raster, ẩn/hiện mode.", PURPLE, 12, 10)
    card(c, 642, 355, 263, 97, "Chia sẻ", "Chạy trong mạng nội bộ và tải dự đoán mẫu dạng JSON.", AMBER, 12, 10)
    c.showPage()


def slide_run(c):
    header(c, 7, "Chạy pipeline và giao diện",
           "Lệnh mẫu trên Windows PowerShell, sau khi đã cài requirements và có checkpoint.")
    rect(c, 55, 142, 850, 224, HexColor("#080D15"), radius=12)
    commands = [
        ("# preprocess", CYAN),
        ("python prerender.py --data-path data/raw/training --output-path data/processed/training --config configs/basic.yaml --n-jobs 1", WHITE),
        ("python prerender.py --data-path data/raw/validation --output-path data/processed/validation --config configs/basic.yaml --n-jobs 1", WHITE),
        ("# train CPU - one epoch smoke run", CYAN),
        ("python train.py --train-data-path data/processed/training --val-data-path", WHITE),
        ("  data/processed/validation --checkpoints-path checkpoints --config", WHITE),
        ("  configs/basic.yaml --device cpu", WHITE),
        ("# launch dashboard", CYAN),
        ("python -m streamlit run app.py --server.address 0.0.0.0", WHITE),
    ]
    for i, (command, color) in enumerate(commands):
        text(c, command, 73, 158 + i * 24, 8.2, color, "Courier")
    card(c, 55, 390, 412, 82, "Mạng nội bộ", "Người cùng mạng mở Network URL do Streamlit hiển thị.", GREEN, 12, 10)
    card(c, 493, 390, 412, 82, "Internet công khai", "Cần máy chủ, checkpoint và cách phân phối dữ liệu phù hợp license.", AMBER, 12, 10)
    c.showPage()


def slide_limits(c):
    header(c, 8, "Giới hạn và hướng tiếp theo",
           "Kết quả trình bày cần gắn với đúng quy mô dữ liệu và thiết bị đã dùng.")
    card(c, 55, 148, 400, 126, "Quy mô dữ liệu", "Một shard mỗi split chỉ phục vụ xác nhận pipeline. Cần nhiều shard/full training trước khi kết luận chất lượng mô hình.", AMBER, 13, 11)
    card(c, 485, 148, 420, 126, "Thiết bị", "Máy phát triển hiện dùng PyTorch CPU. Huấn luyện đầy đủ có thể mất nhiều thời gian; CUDA chưa được xác nhận trên máy này.", ROSE, 13, 11)
    card(c, 55, 298, 400, 126, "Đánh giá", "ADE/FDE hiện là metric chẩn đoán trên split đã prerender, không thay evaluator chính thức của Waymo.", CYAN, 13, 11)
    card(c, 485, 298, 420, 126, "Kế hoạch", "Tăng dữ liệu; lưu lịch sử loss/metric theo epoch; kiểm tra inference; chuẩn bị deployment sau khi xác nhận license và hosting.", PURPLE, 13, 11)
    c.showPage()


def slide_sources(c):
    header(c, 9, "Nguồn và điều khoản", "Ghi nhận ngắn gọn cho phần code và dữ liệu được kế thừa.",
           section="PROJECT PROVENANCE")
    card(c, 55, 145, 850, 118, "Mã nguồn nền",
         "Repository hiện tại là bản điều chỉnh từ MotionCNN Waymo source. Copyright notice, điều khoản và yêu cầu trích dẫn gốc được giữ trong LICENSE.\nhttps://github.com/stepankonev/MotionCNN-Waymo-Open-Motion-Dataset",
         CYAN, 13, 10)
    card(c, 55, 282, 850, 104, "Nghiên cứu liên quan",
         "Stepan Konev, Kirill Brodt, Artsiom Sanakoyeu. MotionCNN: A Strong Baseline for Motion Prediction in Autonomous Driving. 2022.\nhttps://arxiv.org/abs/2206.02163",
         PURPLE, 13, 10)
    card(c, 55, 405, 850, 70, "Dataset",
         "Waymo Open Motion Dataset - dùng theo điều khoản riêng của Waymo: https://waymo.com/open/terms/",
         AMBER, 12, 9)
    c.showPage()


def build():
    c = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("MotionCNN Waymo - Project Implementation")
    c.setAuthor("MotionCNN Waymo project maintainers")
    c.setSubject("Current pipeline, model, evaluation, dashboard, and provenance")
    slides = [slide_cover, slide_scope, slide_data, slide_model, slide_objective,
              slide_dashboard, slide_run, slide_limits, slide_sources]
    for slide in slides:
        slide(c)
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    build()

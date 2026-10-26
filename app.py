
import io
import re
import html
import csv
from datetime import datetime

import streamlit as st
import fitz
import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. CẤU HÌNH ỨNG DỤNG
# =========================================================
st.set_page_config(
    page_title="CVision AI | Trợ lý nghề nghiệp",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #FAF9FF 0%, #F1FBF8 55%, #F4F8FF 100%);
    color: #26324A;
}
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E9E7F5;
}
h1, h2, h3 {
    color: #353452;
    font-weight: 800;
}
div.stButton > button {
    background: linear-gradient(90deg, #8777DF, #6ECDBA);
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 42px;
    font-weight: 700;
}
div.stButton > button:hover {
    color: white;
    border: 1px solid #8777DF;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E9E7F5;
    border-radius: 15px;
    padding: 15px;
}
.hero {
    background: linear-gradient(115deg, #EAE5FF, #DDF7F0, #E4EEFF);
    border-radius: 22px;
    padding: 26px;
    margin-bottom: 20px;
}
.hero h1 { color: #39345F; }
.hero p { color: #535B78; margin-bottom: 0; }
.card {
    background: white;
    border: 1px solid #E8E5F4;
    border-radius: 16px;
    padding: 18px;
    margin: 8px 0;
}
.good {
    display: inline-block;
    background: #DDF7E9;
    color: #176B49;
    padding: 6px 10px;
    border-radius: 18px;
    margin: 3px;
    font-size: 13px;
}
.missing {
    display: inline-block;
    background: #FFE9E5;
    color: #A33C37;
    padding: 6px 10px;
    border-radius: 18px;
    margin: 3px;
    font-size: 13px;
}
.muted { color: #737C91; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. KHỞI TẠO LỊCH SỬ TRONG PHIÊN
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# =========================================================
# 3. MÔ HÌNH AI
# =========================================================
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


# =========================================================
# 4. ĐỌC NỘI DUNG PDF
# =========================================================
def extract_pdf_text(uploaded_file):
    try:
        with fitz.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf"
        ) as pdf:
            text = "\n".join(page.get_text("text") for page in pdf)
        return text.strip()
    except Exception:
        return ""


# =========================================================
# 5. TỪ ĐIỂN KỸ NĂNG
# =========================================================
SKILL_GROUPS = {
    "Python": ["python"],
    "SQL": ["sql", "mysql", "postgresql", "postgres"],
    "Excel": ["excel", "spreadsheet"],
    "Power BI": ["power bi"],
    "Tableau": ["tableau"],
    "Phân tích dữ liệu": ["data analysis", "data analytics", "phân tích dữ liệu"],
    "Trực quan hóa dữ liệu": ["data visualization", "trực quan hóa dữ liệu"],
    "Thống kê": ["statistics", "statistical analysis", "thống kê"],
    "Machine Learning": ["machine learning", "học máy"],
    "Deep Learning": ["deep learning", "học sâu"],
    "NLP": ["nlp", "natural language processing", "xử lý ngôn ngữ tự nhiên"],
    "Trí tuệ nhân tạo": ["artificial intelligence", "trí tuệ nhân tạo"],
    "Marketing": ["marketing", "digital marketing", "tiếp thị"],
    "SEO": ["seo", "search engine optimization"],
    "Google Analytics": ["google analytics", "ga4"],
    "Mạng xã hội": ["social media", "quản lý mạng xã hội"],
    "Content Marketing": ["content marketing", "sáng tạo nội dung"],
    "Giao tiếp": ["communication", "giao tiếp"],
    "Làm việc nhóm": ["teamwork", "team work", "làm việc nhóm"],
    "Quản lý dự án": ["project management", "quản lý dự án"],
    "Quản lý thời gian": ["time management", "quản lý thời gian"],
    "PowerPoint": ["powerpoint"],
    "Microsoft Word": ["microsoft word"],
    "Quản trị kinh doanh": ["business administration", "quản trị kinh doanh"],
    "Tài chính": ["finance", "financial analysis", "tài chính"],
    "Kế toán": ["accounting", "kế toán"],
    "Thiết kế đồ họa": ["graphic design", "thiết kế đồ họa"],
    "Figma": ["figma"],
    "Canva": ["canva"],
    "Java": ["java"],
    "JavaScript": ["javascript"],
    "HTML": ["html"],
    "CSS": ["css"],
    "C++": ["c++"],
    "Git/GitHub": ["github", "git"],
    "Tiếng Anh": ["english", "tiếng anh", "ielts", "toeic"],
    "Nghiên cứu thị trường": ["market research", "nghiên cứu thị trường"],
    "Bán hàng": ["sales", "bán hàng"],
    "Chăm sóc khách hàng": ["customer service", "chăm sóc khách hàng"],
}


def find_skills(text):
    text_lower = text.lower()
    found = []

    for skill, keywords in SKILL_GROUPS.items():
        for keyword in keywords:
            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(keyword.lower())
                + r"(?![a-z0-9])"
            )
            if re.search(pattern, text_lower):
                found.append(skill)
                break

    return found


# =========================================================
# 6. PHÂN TÍCH ĐỘ TƯƠNG ĐỒNG
# =========================================================
def analyze_texts(model, cv_text, job_text):
    vectors = model.encode(
        [cv_text, job_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    similarity = float(
        cosine_similarity(
            vectors[0].reshape(1, -1),
            vectors[1].reshape(1, -1),
        )[0][0]
    )

    # Đưa cosine similarity về thang 0-100 để dễ trình bày.
    # Đây là điểm tương đồng quy đổi, không phải xác suất trúng tuyển.
    score = round(max(0, min(100, similarity * 100)))

    cv_skills = find_skills(cv_text)
    job_skills = find_skills(job_text)
    matched = [s for s in job_skills if s in cv_skills]
    missing = [s for s in job_skills if s not in cv_skills]

    coverage = (
        round(len(matched) / len(job_skills) * 100)
        if job_skills else None
    )

    return {
        "score": score,
        "cv_skills": cv_skills,
        "job_skills": job_skills,
        "matched": matched,
        "missing": missing,
        "coverage": coverage,
    }


# =========================================================
# 7. KIỂM TRA ATS SƠ BỘ
# =========================================================
def check_ats(cv_text):
    text = cv_text.strip()
    lower = text.lower()
    checks = []

    checks.append({
        "name": "Có nội dung văn bản đọc được",
        "ok": len(text) >= 100,
        "detail": "Ứng dụng trích xuất được nội dung PDF."
        if len(text) >= 100
        else "Nội dung trích xuất quá ngắn hoặc PDF có thể là ảnh quét.",
    })

    email_found = bool(
        re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I)
    )
    checks.append({
        "name": "Thông tin email",
        "ok": email_found,
        "detail": "Tìm thấy địa chỉ email."
        if email_found else "Chưa phát hiện email trong nội dung trích xuất.",
    })

    phone_found = bool(
        re.search(r"(\+?\d[\d\s().-]{7,}\d)", text)
    )
    checks.append({
        "name": "Thông tin số điện thoại",
        "ok": phone_found,
        "detail": "Có vẻ đã có số điện thoại."
        if phone_found else "Chưa phát hiện số điện thoại.",
    })

    sections = {
        "Học vấn": ["education", "học vấn", "đào tạo", "university"],
        "Kinh nghiệm": ["experience", "kinh nghiệm", "employment"],
        "Kỹ năng": ["skills", "kỹ năng", "technical skills"],
    }

    for section, keywords in sections.items():
        found = any(k in lower for k in keywords)
        checks.append({
            "name": f"Mục {section}",
            "ok": found,
            "detail": "Đã phát hiện tiêu đề hoặc từ khóa liên quan."
            if found else f"Chưa phát hiện rõ mục {section}.",
        })

    checks.append({
        "name": "Độ dài nội dung",
        "ok": 300 <= len(text) <= 12000,
        "detail": f"Nội dung trích xuất có {len(text)} ký tự. "
        "Đây chỉ là ngưỡng tham khảo, không phải quy chuẩn ATS phổ quát.",
    })

    passed = sum(1 for item in checks if item["ok"])
    return checks, passed, len(checks)


# =========================================================
# 8. GỢI Ý CẢI THIỆN CV
# =========================================================
def make_cv_suggestions(cv_text, job_text, result):
    suggestions = []

    if result["missing"]:
        suggestions.append(
            "Đối chiếu phần kỹ năng còn thiếu với năng lực thực tế. "
            "Chỉ thêm kỹ năng bạn thực sự có và nêu minh chứng cụ thể."
        )

    if not re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", cv_text, re.I):
        suggestions.append(
            "Kiểm tra xem CV đã có email liên hệ chuyên nghiệp hay chưa."
        )

    if not re.search(r"(\+?\d[\d\s().-]{7,}\d)", cv_text):
        suggestions.append(
            "Cân nhắc bổ sung số điện thoại nếu bạn muốn nhà tuyển dụng liên hệ."
        )

    if not any(word in cv_text.lower() for word in [
        "project", "dự án", "portfolio", "sản phẩm"
    ]):
        suggestions.append(
            "Nếu có dự án học tập hoặc cá nhân liên quan, hãy thêm mục dự án "
            "và mô tả vai trò, công cụ, kết quả."
        )

    if result["score"] < 50:
        suggestions.append(
            "Làm rõ các kinh nghiệm và dự án liên quan trực tiếp đến vị trí "
            "đang ứng tuyển; tránh chỉ liệt kê nhiệm vụ chung chung."
        )
    else:
        suggestions.append(
            "Tăng sức thuyết phục bằng các thành tích có số liệu và minh chứng "
            "thực tế nếu bạn có dữ liệu đáng tin cậy."
        )

    suggestions.append(
        "Mẫu diễn đạt tham khảo: 'Thực hiện [nhiệm vụ] bằng [công cụ/phương pháp], "
        "đạt [kết quả có thể kiểm chứng]'. Hãy thay nội dung trong ngoặc bằng "
        "thông tin thật, không tự tạo thành tích."
    )

    return suggestions


# =========================================================
# 9. TẠO CÂU HỎI PHỎNG VẤN
# =========================================================
def make_interview_questions(job_text, cv_text):
    skills = find_skills(job_text)
    cv_skills = find_skills(cv_text)
    questions = [
        (
            "Giới thiệu bản thân trong 60–90 giây.",
            "Nêu nền tảng học tập hoặc kinh nghiệm, thế mạnh liên quan đến vị trí "
            "và lý do bạn quan tâm đến công việc."
        ),
        (
            "Dự án hoặc kinh nghiệm nào thể hiện rõ nhất năng lực của bạn?",
            "Trình bày bối cảnh, vai trò cá nhân, hành động đã thực hiện và kết quả "
            "thực tế. Có thể dùng cấu trúc STAR."
        ),
        (
            "Bạn xử lý thế nào khi gặp một nhiệm vụ mới hoặc vấn đề khó?",
            "Mô tả cách xác định vấn đề, tìm thông tin, thử giải pháp và đánh giá kết quả."
        ),
    ]

    for skill in skills[:5]:
        if skill in cv_skills:
            questions.append((
                f"Bạn đã sử dụng {skill} trong công việc hoặc dự án nào?",
                "Nêu một ví dụ cụ thể, vai trò của bạn, cách sử dụng kỹ năng và kết quả."
            ))
        else:
            questions.append((
                f"Vị trí này yêu cầu {skill}. Bạn sẽ tiếp cận kỹ năng này như thế nào?",
                "Trình bày mức độ hiểu biết hiện tại, kế hoạch học tập và ví dụ thực hành "
                "nếu đã có. Không nên khẳng định kinh nghiệm chưa có."
            ))

    questions.append((
        "Bạn có câu hỏi nào dành cho nhà tuyển dụng không?",
        "Có thể hỏi về mục tiêu của vị trí, tiêu chí đánh giá hiệu quả hoặc cơ hội học hỏi."
    ))

    return questions


# =========================================================
# 10. XUẤT BÁO CÁO PDF TIẾNG VIỆT
# =========================================================
def create_pdf_report(record):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import os

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    font_name = "Helvetica"
    font_bold = "Helvetica-Bold"

    font_candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        (
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ),
    ]

    for regular_path, bold_path in font_candidates:
        if os.path.exists(regular_path) and os.path.exists(bold_path):
            try:
                pdfmetrics.registerFont(TTFont("CVFont", regular_path))
                pdfmetrics.registerFont(TTFont("CVFontBold", bold_path))
                font_name = "CVFont"
                font_bold = "CVFontBold"
                break
            except Exception:
                pass

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CVTitle",
        parent=styles["Title"],
        fontName=font_bold,
        fontSize=19,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#6554C0"),
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="CVHeading",
        parent=styles["Heading2"],
        fontName=font_bold,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#6554C0"),
        spaceBefore=10,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="CVBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9,
        leading=13,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="CVSmall",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#626B80"),
    ))

    def P(value, style="CVBody"):
        safe = html.escape(str(value)).replace("\n", "<br/>")
        return Paragraph(safe, styles[style])

    story = [
        P("CVision AI", "CVTitle"),
        P("BÁO CÁO PHÂN TÍCH HỒ SƠ ỨNG TUYỂN", "CVHeading"),
        P("Thời gian tạo: " + record.get("created_at", ""), "CVSmall"),
        Spacer(1, 10),
        P("1. Tổng quan", "CVHeading"),
        P("Điểm tương đồng tham khảo: "
          + str(record.get("score", 0)) + "/100"),
        P("Điểm số không phải xác suất trúng tuyển.", "CVSmall"),
        P("2. Kỹ năng phù hợp", "CVHeading"),
        P(", ".join(record.get("matched", [])) or "Chưa phát hiện kỹ năng phù hợp."),
        P("3. Kỹ năng cần xem xét bổ sung", "CVHeading"),
        P(", ".join(record.get("missing", [])) or "Không phát hiện kỹ năng còn thiếu trong danh sách từ khóa."),
        P("4. Kiểm tra ATS sơ bộ", "CVHeading"),
    ]

    ats_checks = record.get("ats_checks", [])
    table_data = [["Hạng mục", "Kết quả", "Ghi chú"]]

    for item in ats_checks:
        table_data.append([
            P(item["name"]),
            P("Đạt" if item["ok"] else "Cần xem xét"),
            P(item["detail"]),
        ])

    table = Table(table_data, colWidths=[4.0*cm, 2.4*cm, 10.0*cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAE5FF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#353452")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDD9EE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([table, Spacer(1, 8), P("5. Gợi ý cải thiện", "CVHeading")])

    for item in record.get("suggestions", []):
        story.append(P("• " + item))

    story.extend([
        Spacer(1, 8),
        P(
            "Báo cáo do CVision AI tạo tự động để tham khảo. "
            "Kết quả phụ thuộc vào nội dung PDF, mô tả công việc và từ điển kỹ năng. "
            "Hãy kiểm tra lại thông tin trước khi sử dụng.",
            "CVSmall",
        ),
    ])

    doc.build(story)
    return buffer.getvalue()


# =========================================================
# 11. MENU ĐIỀU HƯỚNG
# =========================================================
with st.sidebar:
    st.markdown("# 📄 CVision AI")
    st.caption("Trợ lý nghề nghiệp bằng AI")
    st.divider()

    page = st.radio(
        "MENU CHÍNH",
        [
            "Tổng quan",
            "Phân tích CV",
            "So sánh vị trí",
            "Gợi ý viết lại CV",
            "Luyện phỏng vấn",
            "Lịch sử phân tích",
            "Hướng dẫn sử dụng",
            "Giới thiệu",
        ],
    )

    st.divider()
    st.caption("Phiên bản hỗ trợ ứng tuyển")


# =========================================================
# 12. TRANG TỔNG QUAN
# =========================================================
if page == "Tổng quan":
    st.markdown("""
    <div class="hero">
      <h1>CVision AI ✨</h1>
      <p>Phân tích CV, đối chiếu vị trí tuyển dụng và chuẩn bị cho hành trình nghề nghiệp.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tính năng chính", "7")
    c2.metric("Mô hình phân tích", "Đa ngôn ngữ")
    c3.metric("API trả phí", "Không bắt buộc")

    st.markdown("### Khám phá các công cụ")
    cards = [
        ("📄", "Phân tích CV", "Đánh giá mức tương đồng, kỹ năng và kiểm tra ATS."),
        ("🎯", "So sánh vị trí", "Đối chiếu một CV với nhiều mô tả tuyển dụng."),
        ("✍️", "Cải thiện CV", "Nhận đề xuất viết rõ ràng và chuyên nghiệp hơn."),
        ("🗣️", "Luyện phỏng vấn", "Chuẩn bị câu hỏi dựa trên vị trí ứng tuyển."),
        ("📊", "Báo cáo PDF", "Tải báo cáo kết quả để lưu hoặc chia sẻ."),
        ("🕘", "Lịch sử", "Xem các kết quả đã phân tích trong phiên hiện tại."),
    ]

    columns = st.columns(2)
    for i, (emoji, title, description) in enumerate(cards):
        with columns[i % 2]:
            st.markdown(
                f'<div class="card"><h3>{emoji} {title}</h3>'
                f'<p>{description}</p></div>',
                unsafe_allow_html=True,
            )

    st.info("Chọn một chức năng trong menu bên trái để bắt đầu.")


# =========================================================
# 13. TRANG PHÂN TÍCH CV
# =========================================================
elif page == "Phân tích CV":
    st.markdown("""
    <div class="hero">
      <h1>Phân tích CV toàn diện 🔎</h1>
      <p>Tải CV PDF và nhập mô tả công việc để xem kết quả.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_cv = st.file_uploader(
        "Tải CV định dạng PDF",
        type=["pdf"],
        key="main_cv",
    )

    cv_text = extract_pdf_text(uploaded_cv) if uploaded_cv else ""

    if uploaded_cv:
        if cv_text:
            st.success(f"Đã trích xuất {len(cv_text)} ký tự từ CV.")
            with st.expander("Xem nội dung đã trích xuất"):
                st.text(cv_text[:10000])
                if len(cv_text) > 10000:
                    st.caption("Nội dung xem trước đã được rút gọn.")
        else:
            st.warning(
                "Không đọc được văn bản từ PDF. "
                "PDF chỉ chứa ảnh có thể cần OCR, hiện chưa được hỗ trợ."
            )

    job_text = st.text_area(
        "Mô tả công việc",
        height=200,
        placeholder="Dán nội dung tin tuyển dụng vào đây...",
        key="main_job",
    )

    if st.button("✨ Phân tích CV ngay", use_container_width=True):
        if not cv_text:
            st.warning("Hãy tải lên CV PDF có văn bản đọc được.")
        elif len(job_text.strip()) < 20:
            st.warning("Hãy nhập mô tả công việc đầy đủ hơn.")
        else:
            try:
                with st.spinner("Mô hình AI đang phân tích..."):
                    model = load_model()
                    result = analyze_texts(model, cv_text, job_text)
                    ats_checks, ats_passed, ats_total = check_ats(cv_text)
                    suggestions = make_cv_suggestions(cv_text, job_text, result)

                    record = {
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "score": result["score"],
                        "matched": result["matched"],
                        "missing": result["missing"],
                        "cv_skills": result["cv_skills"],
                        "job_skills": result["job_skills"],
                        "coverage": result["coverage"],
                        "ats_checks": ats_checks,
                        "ats_passed": ats_passed,
                        "ats_total": ats_total,
                        "suggestions": suggestions,
                        "job_text": job_text,
                        "cv_text": cv_text,
                    }

                    st.session_state.latest_result = record
                    st.session_state.history.insert(0, {
                        "created_at": record["created_at"],
                        "score": record["score"],
                        "job": job_text[:100].replace("\n", " "),
                        "matched_count": len(record["matched"]),
                        "missing_count": len(record["missing"]),
                    })

            except Exception as error:
                st.error("Có lỗi khi phân tích. Vui lòng thử lại.")
                st.exception(error)

    record = st.session_state.latest_result

    if record:
        st.divider()
        st.header("📊 Kết quả phân tích")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Điểm tương đồng", f"{record['score']}/100")
        m2.metric("Kỹ năng phù hợp", len(record["matched"]))
        m3.metric("Kỹ năng còn thiếu", len(record["missing"]))
        m4.metric("Kiểm tra ATS", f"{record['ats_passed']}/{record['ats_total']}")

        st.caption(
            "Điểm tương đồng được quy đổi từ cosine similarity; "
            "không phải xác suất trúng tuyển."
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=record["score"],
            number={"suffix": "/100"},
            title={"text": "Mức độ tương đồng ngữ nghĩa"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#8272D8"},
                "steps": [
                    {"range": [0, 40], "color": "#FFE5E1"},
                    {"range": [40, 70], "color": "#FFF0D7"},
                    {"range": [70, 100], "color": "#DDF7E9"},
                ],
            },
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=15, r=15, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.subheader("✅ Kỹ năng phù hợp")
            if record["matched"]:
                st.markdown(
                    "".join(
                        f'<span class="good">{html.escape(s)}</span>'
                        for s in record["matched"]
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.info("Chưa phát hiện kỹ năng trùng khớp trong từ điển hiện tại.")

        with right:
            st.subheader("📌 Kỹ năng cần xem xét")
            if record["missing"]:
                st.markdown(
                    "".join(
                        f'<span class="missing">{html.escape(s)}</span>'
                        for s in record["missing"]
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.success("Không phát hiện kỹ năng còn thiếu trong từ điển hiện tại.")

        if record["coverage"] is not None:
            st.subheader("Mức độ bao phủ kỹ năng")
            st.progress(record["coverage"] / 100)
            st.write(
                f"Phát hiện {len(record['matched'])}/"
                f"{len(record['job_skills'])} kỹ năng từ mô tả công việc "
                f"({record['coverage']}%)."
            )

        st.subheader("🧾 Kiểm tra ATS sơ bộ")
        for item in record["ats_checks"]:
            if item["ok"]:
                st.success(f"✓ {item['name']}: {item['detail']}")
            else:
                st.warning(f"• {item['name']}: {item['detail']}")

        st.subheader("💡 Gợi ý cải thiện")
        for suggestion in record["suggestions"]:
            st.markdown(f"- {suggestion}")

        try:
            pdf_bytes = create_pdf_report(record)
            st.download_button(
                "📥 Tải báo cáo phân tích PDF",
                data=pdf_bytes,
                file_name="CVisionAI_bao_cao.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as error:
            st.warning(
                "Chưa tạo được báo cáo PDF. Kiểm tra thư viện reportlab "
                "và font tiếng Việt trong môi trường triển khai."
            )


# =========================================================
# 14. SO SÁNH MỘT CV VỚI NHIỀU VỊ TRÍ
# =========================================================
elif page == "So sánh vị trí":
    st.markdown("""
    <div class="hero">
      <h1>So sánh nhiều vị trí tuyển dụng 🎯</h1>
      <p>Đánh giá một CV với nhiều mô tả công việc để xem mức tương đồng tham khảo.</p>
    </div>
    """, unsafe_allow_html=True)

    compare_file = st.file_uploader(
        "Tải CV PDF",
        type=["pdf"],
        key="compare_cv",
    )
    compare_cv_text = extract_pdf_text(compare_file) if compare_file else ""

    jobs_input = st.text_area(
        "Nhập các mô tả công việc",
        height=300,
        placeholder=(
            "Vị trí 1: Marketing Intern...\n"
            "---\n"
            "Vị trí 2: Data Analyst Intern...\n"
            "---\n"
            "Vị trí 3: Business Analyst Intern..."
        ),
    )
    st.caption("Ngăn cách các mô tả công việc bằng một dòng chứa ba dấu gạch: ---")

    if st.button("📊 So sánh các vị trí", use_container_width=True):
        jobs = [
            j.strip() for j in re.split(r"\n\s*---+\s*\n", jobs_input)
            if len(j.strip()) >= 20
        ]

        if not compare_cv_text:
            st.warning("Vui lòng tải lên CV PDF có văn bản đọc được.")
        elif len(jobs) < 2:
            st.warning("Hãy nhập ít nhất hai mô tả công việc, ngăn cách bằng ---.")
        else:
            try:
                with st.spinner("Đang so sánh các vị trí..."):
                    model = load_model()
                    comparison = []
                    for index, job in enumerate(jobs, start=1):
                        result = analyze_texts(model, compare_cv_text, job)
                        comparison.append({
                            "Vị trí": f"Vị trí {index}",
                            "Điểm tương đồng": result["score"],
                            "Kỹ năng phù hợp": len(result["matched"]),
                            "Kỹ năng cần xem xét": len(result["missing"]),
                            "Mô tả": job,
                        })

                comparison.sort(
                    key=lambda x: x["Điểm tương đồng"],
                    reverse=True,
                )
                st.session_state["comparison_results"] = comparison

            except Exception as error:
                st.error("Không thể so sánh các vị trí.")
                st.exception(error)

    comparison = st.session_state.get("comparison_results", [])

    if comparison:
        st.subheader("Bảng xếp hạng mức tương đồng")
        st.dataframe(
            [
                {
                    "Xếp hạng": index,
                    "Vị trí": item["Vị trí"],
                    "Điểm tương đồng": f"{item['Điểm tương đồng']}/100",
                    "Kỹ năng phù hợp": item["Kỹ năng phù hợp"],
                    "Kỹ năng cần xem xét": item["Kỹ năng cần xem xét"],
                }
                for index, item in enumerate(comparison, start=1)
            ],
            use_container_width=True,
            hide_index=True,
        )

        fig = go.Figure(go.Bar(
            x=[x["Điểm tương đồng"] for x in comparison],
            y=[x["Vị trí"] for x in comparison],
            orientation="h",
            text=[f"{x['Điểm tương đồng']}/100" for x in comparison],
            textposition="auto",
        ))
        fig.update_layout(
            title="So sánh điểm tương đồng",
            xaxis_title="Điểm quy đổi (0–100)",
            yaxis_title="",
            xaxis=dict(range=[0, 100]),
            height=max(300, 80 * len(comparison)),
        )
        st.plotly_chart(fig, use_container_width=True)

        selected = st.selectbox(
            "Xem chi tiết mô tả công việc",
            options=list(range(len(comparison))),
            format_func=lambda i: (
                f"{comparison[i]['Vị trí']} — "
                f"{comparison[i]['Điểm tương đồng']}/100"
            ),
        )
        st.text_area(
            "Nội dung mô tả",
            value=comparison[selected]["Mô tả"],
            height=180,
            disabled=True,
        )

        st.caption(
            "Thứ hạng chỉ phản ánh độ tương đồng ngữ nghĩa theo mô hình; "
            "không bảo đảm vị trí đó phù hợp nhất về mọi mặt."
        )


# =========================================================
# 15. GỢI Ý VIẾT LẠI CV
# =========================================================
elif page == "Gợi ý viết lại CV":
    st.markdown("""
    <div class="hero">
      <h1>Trợ lý cải thiện CV ✍️</h1>
      <p>Nhận gợi ý diễn đạt dựa trên thông tin bạn cung cấp, không tự tạo thành tích.</p>
    </div>
    """, unsafe_allow_html=True)

    rewrite_file = st.file_uploader(
        "Tải CV PDF",
        type=["pdf"],
        key="rewrite_cv",
    )
    rewrite_cv_text = extract_pdf_text(rewrite_file) if rewrite_file else ""

    rewrite_job = st.text_area(
        "Mô tả công việc mục tiêu",
        height=160,
        placeholder="Dán mô tả công việc để nhận gợi ý phù hợp...",
    )

    if rewrite_cv_text:
        st.success("Đã trích xuất nội dung CV.")
        with st.expander("Xem nội dung CV"):
            st.text(rewrite_cv_text[:10000])

    if st.button("✨ Tạo gợi ý cải thiện", use_container_width=True):
        if not rewrite_cv_text:
            st.warning("Vui lòng tải CV PDF có văn bản đọc được.")
        elif len(rewrite_job.strip()) < 20:
            st.warning("Vui lòng nhập mô tả công việc.")
        else:
            try:
                model = load_model()
                result = analyze_texts(model, rewrite_cv_text, rewrite_job)
                suggestions = make_cv_suggestions(
                    rewrite_cv_text, rewrite_job, result
                )

                st.session_state["rewrite_output"] = {
                    "suggestions": suggestions,
                    "cv_text": rewrite_cv_text,
                    "job_text": rewrite_job,
                    "result": result,
                }
            except Exception as error:
                st.error("Không thể tạo gợi ý lúc này.")
                st.exception(error)

    output = st.session_state.get("rewrite_output")

    if output:
        st.subheader("1. Nội dung cần xem xét")
        for suggestion in output["suggestions"]:
            st.markdown(f"- {suggestion}")

        st.subheader("2. Mẫu viết lại phần giới thiệu")
        st.info(
            "Mẫu tham khảo: “Tôi có nền tảng về [lĩnh vực/kỹ năng thực tế], "
            "đã vận dụng trong [dự án hoặc trải nghiệm cụ thể]. Tôi mong muốn "
            "phát triển năng lực [mục tiêu liên quan] và đóng góp vào "
            "[nhiệm vụ phù hợp với vị trí].”"
        )
        st.caption(
            "Thay phần trong ngoặc bằng thông tin có thật. Đây là mẫu hướng dẫn "
            "theo quy tắc, chưa phải mô hình sinh văn bản tự động."
        )

        st.subheader("3. Mẫu diễn đạt kinh nghiệm")
        st.code(
            "Đã thực hiện [nhiệm vụ cụ thể] bằng [công cụ/phương pháp], "
            "nhằm [mục tiêu] và đạt [kết quả có thể kiểm chứng].",
            language="text",
        )

        st.download_button(
            "📥 Tải gợi ý dạng TXT",
            data="\n".join(output["suggestions"]),
            file_name="CVisionAI_goi_y_CV.txt",
            mime="text/plain",
        )


# =========================================================
# 16. LUYỆN PHỎNG VẤN
# =========================================================
elif page == "Luyện phỏng vấn":
    st.markdown("""
    <div class="hero">
      <h1>Luyện phỏng vấn 🗣️</h1>
      <p>Chuẩn bị câu hỏi theo vị trí tuyển dụng và các kỹ năng được yêu cầu.</p>
    </div>
    """, unsafe_allow_html=True)

    interview_file = st.file_uploader(
        "Tải CV PDF (không bắt buộc)",
        type=["pdf"],
        key="interview_cv",
    )
    interview_cv_text = extract_pdf_text(interview_file) if interview_file else ""

    interview_job = st.text_area(
        "Mô tả vị trí ứng tuyển",
        height=170,
        placeholder="Nhập yêu cầu của vị trí...",
    )

    if st.button("🎤 Tạo bộ câu hỏi phỏng vấn", use_container_width=True):
        if len(interview_job.strip()) < 20:
            st.warning("Hãy nhập mô tả vị trí ứng tuyển.")
        else:
            questions = make_interview_questions(
                interview_job, interview_cv_text
            )
            st.session_state["interview_questions"] = questions

    questions = st.session_state.get("interview_questions", [])

    if questions:
        st.subheader("Bộ câu hỏi dành cho bạn")

        for index, (question, hint) in enumerate(questions, start=1):
            with st.expander(f"Câu {index}: {question}", expanded=(index == 1)):
                st.markdown("**Gợi ý chuẩn bị câu trả lời**")
                st.write(hint)

                answer = st.text_area(
                    "Câu trả lời của bạn",
                    key=f"interview_answer_{index}",
                    height=120,
                    placeholder="Hãy thử trả lời bằng lời của chính bạn...",
                )

                if answer.strip():
                    st.caption(
                        f"Bạn đã nhập {len(answer.split())} từ. "
                        "Hãy kiểm tra câu trả lời có ví dụ thực tế, vai trò cá nhân "
                        "và kết quả rõ ràng hay chưa."
                    )

        st.download_button(
            "📥 Tải bộ câu hỏi",
            data="\n\n".join(
                f"Câu {i}: {q}\nGợi ý: {hint}"
                for i, (q, hint) in enumerate(questions, start=1)
            ),
            file_name="CVisionAI_cau_hoi_phong_van.txt",
            mime="text/plain",
        )

        st.caption(
            "Phiên bản này tạo câu hỏi và gợi ý theo mẫu; chưa có mô hình chấm điểm "
            "câu trả lời tự do hoặc hội thoại phỏng vấn thời gian thực."
        )


# =========================================================
# 17. LỊCH SỬ PHÂN TÍCH
# =========================================================
elif page == "Lịch sử phân tích":
    st.markdown("""
    <div class="hero">
      <h1>Lịch sử phân tích 🕘</h1>
      <p>Các kết quả được ghi lại trong phiên sử dụng hiện tại.</p>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.history

    if history:
        st.dataframe(
            [
                {
                    "Thời gian": item["created_at"],
                    "Mô tả công việc": item["job"],
                    "Điểm tương đồng": f"{item['score']}/100",
                    "Kỹ năng phù hợp": item["matched_count"],
                    "Kỹ năng cần xem xét": item["missing_count"],
                }
                for item in history
            ],
            use_container_width=True,
            hide_index=True,
        )

        csv_buffer = io.StringIO()
        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=[
                "created_at", "score", "job",
                "matched_count", "missing_count"
            ],
        )
        writer.writeheader()
        writer.writerows(history)

        st.download_button(
            "📥 Tải lịch sử dạng CSV",
            data=csv_buffer.getvalue().encode("utf-8-sig"),
            file_name="CVisionAI_lich_su.csv",
            mime="text/csv",
        )

        if st.button("Xóa lịch sử trong phiên"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info(
            "Chưa có lịch sử. Hãy thực hiện phân tích ở mục Phân tích CV trước."
        )

    st.caption(
        "Lịch sử này chỉ tồn tại trong phiên ứng dụng hiện tại. "
        "Tải CSV về máy nếu muốn lưu lâu dài."
    )


# =========================================================
# 18. HƯỚNG DẪN
# =========================================================
elif page == "Hướng dẫn sử dụng":
    st.title("📘 Hướng dẫn sử dụng")

    st.markdown("""
    1. **Phân tích CV:** tải tệp PDF có văn bản và nhập mô tả công việc.
    2. **Kiểm tra ATS:** xem các mục thông tin liên hệ, học vấn, kinh nghiệm,
       kỹ năng và khả năng trích xuất văn bản.
    3. **So sánh vị trí:** nhập từ hai mô tả công việc trở lên, phân cách bằng `---`.
    4. **Gợi ý viết lại CV:** tham khảo mẫu diễn đạt và chỉ dùng thông tin có thật.
    5. **Luyện phỏng vấn:** tạo câu hỏi theo mô tả vị trí, thử nhập câu trả lời.
    6. **Báo cáo PDF:** tải báo cáo trong trang Phân tích CV.
    7. **Lịch sử:** xem hoặc tải CSV trong phiên hiện tại.

    **Giới hạn hiện tại**
    - PDF chỉ chứa ảnh có thể không trích xuất được nếu chưa dùng OCR.
    - Kiểm tra ATS là kiểm tra sơ bộ dựa trên văn bản, không phải công cụ ATS thương mại.
    - Nhận diện kỹ năng phụ thuộc vào danh sách từ khóa.
    - Điểm tương đồng không phải xác suất trúng tuyển.
    - Các gợi ý viết lại và câu hỏi phỏng vấn dựa trên mẫu, không phải nội dung do mô hình ngôn ngữ sinh ra.
    """)


# =========================================================
# 19. GIỚI THIỆU
# =========================================================
elif page == "Giới thiệu":
    st.markdown("""
    <div class="hero">
      <h1>Về CVision AI</h1>
      <p>Công cụ hỗ trợ phân tích hồ sơ và chuẩn bị ứng tuyển.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Công nghệ sử dụng
    - **Streamlit:** giao diện ứng dụng web.
    - **Sentence Transformers:** biểu diễn và so sánh ngữ nghĩa văn bản.
    - **Scikit-learn:** tính độ tương đồng cosine.
    - **PyMuPDF:** trích xuất văn bản từ PDF.
    - **Plotly:** biểu đồ kết quả.
    - **ReportLab:** tạo báo cáo PDF.

    ### Nguyên tắc sử dụng
    CVision AI hỗ trợ tham khảo, không quyết định thay nhà tuyển dụng.
    Người dùng cần xác minh mọi thông tin và không nên thêm kỹ năng hoặc thành tích
    không có thật vào CV.
    """)

st.divider()
st.markdown(
    '<p class="muted" style="text-align:center">'
    'CVision AI · Trợ lý hỗ trợ ứng tuyển</p>',
    unsafe_allow_html=True,
)

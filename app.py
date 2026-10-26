
import streamlit as st
import fitz
import pytesseract
import io
import re
import pandas as pd
import plotly.express as px

from PIL import Image
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. CẤU HÌNH ỨNG DỤNG
# =========================================================

st.set_page_config(
    page_title="CVision AI | Career Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. GIAO DIỆN XANH NAVY - XANH DƯƠNG - TRẮNG
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #102A43;
    --navy-light: #173F75;
    --blue: #2563EB;
    --blue-light: #60A5FA;
    --page: #F3F7FC;
    --text: #172B4D;
    --muted: #64748B;
    --border: #E1EAF5;
}

html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif;
}

.stApp {
    background: var(--page);
    color: var(--text);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102A43 0%, #173F75 100%);
    border-right: 1px solid #244A72;
}

[data-testid="stSidebar"] * {
    color: #F8FAFC;
}

[data-testid="stSidebar"] hr {
    border-color: #345779;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 7px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent;
    border-radius: 10px;
    padding: 10px 12px;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #244A72;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #2563EB;
    border: 1px solid #60A5FA;
}

h1, h2, h3 {
    color: var(--navy);
    font-weight: 800;
    letter-spacing: -0.4px;
}

h1 { font-size: clamp(1.8rem, 3vw, 2.6rem); }
h2 { font-size: 1.55rem; }
h3 { font-size: 1.15rem; }

.hero {
    background: linear-gradient(120deg, #102A43 0%, #174B83 65%, #2563EB 100%);
    border: 1px solid #2C5B91;
    border-radius: 22px;
    padding: 30px 32px;
    margin-bottom: 26px;
    box-shadow: 0 12px 32px rgba(16, 42, 67, 0.12);
}

.hero h1 {
    color: #FFFFFF !important;
    margin-bottom: 8px;
}

.hero p {
    color: #DCEBFA !important;
    font-size: 15px;
    line-height: 1.8;
}

.card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 22px;
    margin: 8px 0;
    box-shadow: 0 4px 14px rgba(16, 42, 67, 0.035);
}

.card h3 {
    margin-top: 0;
    color: var(--navy);
}

.card p {
    color: var(--muted);
    line-height: 1.7;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 4px 14px rgba(16, 42, 67, 0.035);
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: var(--navy);
    font-weight: 800;
}

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(100deg, #1D4ED8, #2563EB);
    color: #FFFFFF !important;
    border: 1px solid #2563EB;
    border-radius: 11px;
    padding: 0.65rem 1.1rem;
    min-height: 44px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: #173F75;
    border-color: #173F75;
}

.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {
    background: #FFFFFF;
    border: 1px solid #D5E1F0;
    border-radius: 10px;
    color: var(--text);
}

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #8CB6EA;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 13px;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

.good {
    display: inline-block;
    background: #DBEAFE;
    color: #174A8B;
    border: 1px solid #BFDBFE;
    padding: 6px 11px;
    border-radius: 8px;
    margin: 3px;
    font-size: 13px;
    font-weight: 600;
}

.missing {
    display: inline-block;
    background: #FFF1F2;
    color: #B42336;
    border: 1px solid #FECDD3;
    padding: 6px 11px;
    border-radius: 8px;
    margin: 3px;
    font-size: 13px;
    font-weight: 600;
}

.muted {
    color: var(--muted);
    font-size: 13px;
}

hr {
    border-color: var(--border);
}

@media (max-width: 768px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1rem;
    }

    .hero {
        padding: 22px 20px;
        border-radius: 16px;
    }

    [data-testid="stMetric"] {
        padding: 12px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DỮ LIỆU PHIÊN LÀM VIỆC
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "compare_results" not in st.session_state:
    st.session_state.compare_results = None

if "rewrite_output" not in st.session_state:
    st.session_state.rewrite_output = ""

if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = []


# =========================================================
# 4. TẢI MÔ HÌNH AI
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def calculate_similarity(cv_text, job_text):
    model = load_model()

    embeddings = model.encode(
        [cv_text, job_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    score = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0][0]

    # Đưa kết quả cosine về khoảng 0-100 để trình bày.
    # Đây không phải xác suất trúng tuyển.
    return max(0.0, min(1.0, float(score))) * 100


# =========================================================
# 5. OCR: NHẬN PDF, PDF SCAN, JPG, JPEG, PNG
# =========================================================

def run_ocr(image):
    """Ưu tiên OCR tiếng Việt và tiếng Anh."""
    image = image.convert("RGB")

    try:
        return pytesseract.image_to_string(
            image,
            lang="vie+eng"
        ).strip()
    except pytesseract.TesseractError:
        # Dự phòng nếu bộ ngôn ngữ tiếng Việt chưa được cài.
        try:
            return pytesseract.image_to_string(
                image,
                lang="eng"
            ).strip()
        except Exception as e:
            raise RuntimeError(
                "OCR không hoạt động. Hãy kiểm tra Tesseract OCR "
                "và các bộ ngôn ngữ đã được cài đặt."
            ) from e


def extract_cv_text(uploaded_file):
    """
    Đọc CV từ PDF, JPG, JPEG hoặc PNG.
    PDF có văn bản sẽ được đọc trực tiếp.
    PDF scan và ảnh sẽ được xử lý bằng OCR.
    """
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    try:
        if filename.endswith(".pdf"):
            document = fitz.open(
                stream=file_bytes,
                filetype="pdf"
            )

            extracted_pages = []

            for page in document:
                page_text = page.get_text("text").strip()

                # Nếu trang PDF không có lớp văn bản,
                # chuyển trang thành ảnh để OCR.
                if len(page_text) < 20:
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(2, 2),
                        alpha=False
                    )

                    page_image = Image.open(
                        io.BytesIO(pix.tobytes("png"))
                    )

                    page_text = run_ocr(page_image)

                extracted_pages.append(page_text)

            document.close()
            return "\n".join(extracted_pages).strip()

        elif filename.endswith((".jpg", ".jpeg", ".png")):
            image = Image.open(
                io.BytesIO(file_bytes)
            )

            return run_ocr(image)

        else:
            raise ValueError(
                "Định dạng không được hỗ trợ. "
                "Vui lòng tải PDF, JPG, JPEG hoặc PNG."
            )

    except Exception as e:
        raise RuntimeError(
            f"Không thể đọc CV: {e}"
        ) from e


# =========================================================
# 6. PHÂN TÍCH KỸ NĂNG
# =========================================================

SKILL_LIST = [
    "digital marketing",
    "content marketing",
    "market research",
    "project management",
    "data analysis",
    "customer service",
    "problem solving",
    "social media",
    "email marketing",
    "communication",
    "teamwork",
    "leadership",
    "marketing",
    "sales",
    "seo",
    "sem",
    "google ads",
    "facebook ads",
    "copywriting",
    "branding",
    "excel",
    "power bi",
    "tableau",
    "python",
    "sql",
    "canva",
    "photoshop",
    "graphic design",
    "english",
    "tiếng anh",
    "phân tích dữ liệu",
    "giao tiếp",
    "làm việc nhóm",
    "quản lý dự án",
    "nghiên cứu thị trường",
    "sáng tạo nội dung",
    "quảng cáo",
    "thương hiệu",
    "kinh doanh",
    "bán hàng",
    "thuyết trình",
    "tin học văn phòng",
    "word",
    "powerpoint"
]


def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def find_skills(text):
    normalized = normalize_text(text)
    found = []

    for skill in SKILL_LIST:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, normalized):
            found.append(skill)

    return sorted(set(found))


def get_match_level(score):
    if score >= 75:
        return "Mức độ tương đồng cao"
    elif score >= 50:
        return "Mức độ tương đồng trung bình"

    return "Mức độ tương đồng thấp"


# =========================================================
# 7. CÁC HÀM GIAO DIỆN DÙNG CHUNG
# =========================================================

def show_hero(title, description):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_upload_help():
    st.caption(
        "Hỗ trợ PDF có văn bản, PDF scan, JPG, JPEG và PNG. "
        "Ảnh mờ hoặc chữ viết tay có thể không được OCR chính xác."
    )


def upload_cv(key):
    uploaded = st.file_uploader(
        "Tải CV của ứng viên",
        type=["pdf", "jpg", "jpeg", "png"],
        key=key,
        help="Hỗ trợ PDF, JPG, JPEG và PNG."
    )

    show_upload_help()
    return uploaded


# =========================================================
# 8. MENU ĐIỀU HƯỚNG
# =========================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 18px 0;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
            <div style="
                background:#2563EB;
                border:1px solid #60A5FA;
                width:46px;
                height:46px;
                border-radius:13px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:24px;
            ">📄</div>

            <div>
                <div style="font-size:22px;font-weight:800;color:#FFFFFF;">
                    CVision AI
                </div>
                <div style="font-size:11px;color:#C5D9EF;">
                    CAREER INTELLIGENCE
                </div>
            </div>
        </div>

        <div style="height:1px;background:#345779;margin:12px 0;"></div>

        <div style="font-size:12px;color:#D6E5F5;line-height:1.8;">
            Phân tích hồ sơ ứng tuyển<br>
            <span style="color:#8FC1FF;">
                AI-POWERED CAREER ASSISTANT
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "ĐIỀU HƯỚNG",
        [
            "Tổng quan",
            "Phân tích CV",
            "So sánh vị trí",
            "Gợi ý viết lại CV",
            "Luyện phỏng vấn",
            "Lịch sử phân tích",
            "Hướng dẫn sử dụng",
            "Giới thiệu"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.caption("CVISION AI · CAREER PLATFORM")


# =========================================================
# 9. TRANG TỔNG QUAN
# =========================================================

if page == "Tổng quan":
    show_hero(
        "CVision AI",
        "Nền tảng hỗ trợ phân tích hồ sơ ứng tuyển, so sánh CV với "
        "mô tả công việc và định hướng cải thiện hồ sơ bằng AI."
    )

    st.subheader("Chào mừng đến với CVision AI")

    st.write(
        "Tải CV lên, nhập mô tả công việc và xem những điểm cần "
        "đối chiếu trước khi ứng tuyển."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
            <h3>📄 Phân tích CV</h3>
            <p>Đọc hồ sơ PDF hoặc ảnh và so sánh ngữ nghĩa với yêu cầu tuyển dụng.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <h3>🎯 So sánh vị trí</h3>
            <p>Đối chiếu một CV với nhiều mô tả công việc để tham khảo lựa chọn.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
            <h3>✨ Phát triển hồ sơ</h3>
            <p>Rà soát kỹ năng, cải thiện cách diễn đạt và luyện phỏng vấn.</p>
        </div>
        """, unsafe_allow_html=True)

    history = st.session_state.history

    if history:
        average_score = sum(
            item["Điểm tương đồng (%)"] for item in history
        ) / len(history)

        highest_score = max(
            item["Điểm tương đồng (%)"] for item in history
        )
    else:
        average_score = 0
        highest_score = 0

    st.subheader("Hoạt động trong phiên hiện tại")

    m1, m2, m3 = st.columns(3)
    m1.metric("Lượt phân tích", len(history))
    m2.metric("Điểm trung bình", f"{average_score:.1f}%")
    m3.metric("Điểm cao nhất", f"{highest_score:.1f}%")

    st.info(
        "Điểm tương đồng phản ánh mức độ gần nhau của hai văn bản "
        "theo mô hình AI, không phải xác suất được tuyển dụng."
    )


# =========================================================
# 10. TRANG PHÂN TÍCH CV
# =========================================================

elif page == "Phân tích CV":
    show_hero(
        "Phân tích CV",
        "Tải hồ sơ PDF hoặc ảnh, nhập mô tả công việc và kiểm tra "
        "mức độ tương đồng cùng các từ khóa kỹ năng."
    )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("01 · Hồ sơ ứng viên")
        uploaded_cv = upload_cv("analysis_cv")

        if uploaded_cv:
            st.caption(f"File đã chọn: {uploaded_cv.name}")

    with right:
        st.subheader("02 · Vị trí ứng tuyển")

        job_title = st.text_input(
            "Tên vị trí",
            placeholder="Ví dụ: Marketing Intern",
            key="analysis_job_title"
        )

        job_description = st.text_area(
            "Mô tả công việc (Job Description)",
            placeholder="Dán nhiệm vụ, yêu cầu kỹ năng và tiêu chí tuyển dụng...",
            height=220,
            key="analysis_job_description"
        )

    if st.button(
        "🚀 Phân tích mức độ phù hợp",
        use_container_width=True,
        key="run_analysis"
    ):
        if not uploaded_cv:
            st.error("Vui lòng tải CV trước khi phân tích.")

        elif not job_description.strip():
            st.error("Vui lòng nhập mô tả công việc.")

        else:
            try:
                with st.spinner(
                    "Đang đọc CV và xử lý bằng AI. "
                    "Nếu là ảnh hoặc PDF scan, OCR có thể mất thêm thời gian..."
                ):
                    cv_text = extract_cv_text(uploaded_cv)

                    if len(cv_text.strip()) < 30:
                        st.error(
                            "Không trích xuất được đủ văn bản từ CV. "
                            "Hãy thử ảnh rõ nét hơn hoặc PDF có thể chọn văn bản."
                        )
                    else:
                        score = calculate_similarity(
                            cv_text,
                            job_description
                        )

                        cv_skills = find_skills(cv_text)
                        job_skills = find_skills(job_description)

                        matched_skills = [
                            skill for skill in job_skills
                            if skill in cv_skills
                        ]

                        missing_skills = [
                            skill for skill in job_skills
                            if skill not in cv_skills
                        ]

                        result = {
                            "filename": uploaded_cv.name,
                            "job_title": (
                                job_title.strip()
                                or "Chưa đặt tên vị trí"
                            ),
                            "score": score,
                            "cv_text": cv_text,
                            "job_text": job_description,
                            "cv_skills": cv_skills,
                            "job_skills": job_skills,
                            "matched_skills": matched_skills,
                            "missing_skills": missing_skills
                        }

                        st.session_state.analysis_result = result

                        st.session_state.history.append({
                            "Thời gian": datetime.now().strftime(
                                "%d/%m/%Y %H:%M"
                            ),
                            "Tên CV": uploaded_cv.name,
                            "Vị trí": result["job_title"],
                            "Điểm tương đồng (%)": round(score, 1),
                            "Kỹ năng trùng khớp": len(matched_skills),
                            "Kỹ năng cần xem xét": len(missing_skills)
                        })

            except Exception as e:
                st.error(f"Không thể phân tích CV: {e}")

    result = st.session_state.analysis_result

    if result:
        st.divider()
        st.subheader("Kết quả phân tích")

        m1, m2, m3 = st.columns(3)

        m1.metric("Điểm tương đồng", f"{result['score']:.1f}%")
        m2.metric(
            "Kỹ năng trùng khớp",
            len(result["matched_skills"])
        )
        m3.metric(
            "Kỹ năng cần xem xét",
            len(result["missing_skills"])
        )

        st.progress(int(result["score"]) / 100)

        st.markdown(
            f"**Đánh giá tham khảo:** {get_match_level(result['score'])}"
        )

        st.caption(
            "Điểm tương đồng không phải điểm ATS chính thức hay xác suất "
            "được tuyển dụng. Kết quả từ khóa phụ thuộc vào danh sách kỹ năng "
            "được thiết lập."
        )

        left, right = st.columns(2)

        with left:
            st.markdown("### ✅ Kỹ năng trùng khớp")

            if result["matched_skills"]:
                html = " ".join(
                    f'<span class="good">{skill}</span>'
                    for skill in result["matched_skills"]
                )
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.write(
                    "Chưa phát hiện kỹ năng trùng khớp trong danh sách "
                    "từ khóa đã cài đặt."
                )

        with right:
            st.markdown("### 🔎 Kỹ năng cần đối chiếu")

            if result["missing_skills"]:
                html = " ".join(
                    f'<span class="missing">{skill}</span>'
                    for skill in result["missing_skills"]
                )
                st.markdown(html, unsafe_allow_html=True)

                st.caption(
                    "Kiểm tra xem bạn có kỹ năng này nhưng chưa thể hiện "
                    "trong CV hay không."
                )
            else:
                st.write(
                    "Không phát hiện từ khóa còn thiếu trong danh sách "
                    "kỹ năng đã cài đặt."
                )

        if result["job_skills"]:
            chart_df = pd.DataFrame({
                "Nhóm": [
                    "Đã tìm thấy trong CV",
                    "Chưa tìm thấy trong CV"
                ],
                "Số lượng": [
                    len(result["matched_skills"]),
                    len(result["missing_skills"])
                ]
            })

            fig = px.bar(
                chart_df,
                x="Nhóm",
                y="Số lượng",
                text="Số lượng",
                title="Đối chiếu từ khóa kỹ năng"
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color="#172B4D"),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📝 Gợi ý cải thiện")

        suggestions = []

        if result["score"] < 50:
            suggestions.append(
                "Kiểm tra lại mức độ liên quan giữa CV và vị trí ứng tuyển."
            )

        if result["missing_skills"]:
            suggestions.append(
                "Đối chiếu các kỹ năng còn thiếu với năng lực và kinh nghiệm "
                "thực tế trước khi cập nhật CV."
            )

        if not re.search(
            r"\b\d+%|\b\d+\s*(năm|tháng|dự án|khách hàng|người)\b",
            result["cv_text"].lower()
        ):
            suggestions.append(
                "Nếu có căn cứ, hãy bổ sung kết quả cụ thể bằng số liệu "
                "trong phần kinh nghiệm hoặc dự án."
            )

        suggestions.append(
            "Ưu tiên kinh nghiệm, dự án và kỹ năng liên quan trực tiếp "
            "đến mô tả công việc."
        )

        suggestions.append(
            "Không thêm kỹ năng hoặc thành tích mà bạn không thực sự có."
        )

        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

        report = f"""BÁO CÁO PHÂN TÍCH CVISION AI

Tên CV: {result['filename']}
Vị trí ứng tuyển: {result['job_title']}
Điểm tương đồng: {result['score']:.1f}%
Đánh giá: {get_match_level(result['score'])}

KỸ NĂNG TRÙNG KHỚP:
{', '.join(result['matched_skills']) or 'Chưa phát hiện'}

KỸ NĂNG TRONG JD CHƯA THẤY TRONG CV:
{', '.join(result['missing_skills']) or 'Chưa phát hiện'}

LƯU Ý:
Điểm tương đồng là chỉ số tham khảo, không phải xác suất được tuyển dụng.
Kết quả từ khóa phụ thuộc vào danh sách kỹ năng được thiết lập.
"""

        st.download_button(
            "⬇️ Tải báo cáo phân tích",
            data=report.encode("utf-8"),
            file_name="CVisionAI_bao_cao.txt",
            mime="text/plain"
        )


# =========================================================
# 11. TRANG SO SÁNH VỊ TRÍ
# =========================================================

elif page == "So sánh vị trí":
    show_hero(
        "So sánh vị trí ứng tuyển",
        "Đối chiếu một CV với nhiều mô tả công việc để tham khảo "
        "mức độ tương đồng của từng vị trí."
    )

    uploaded_cv = upload_cv("compare_cv")

    job1 = st.text_area(
        "Vị trí 1 · Tên và mô tả công việc",
        height=130,
        placeholder="Dòng đầu ghi tên vị trí, bên dưới là mô tả công việc..."
    )

    job2 = st.text_area(
        "Vị trí 2 · Tên và mô tả công việc",
        height=130,
        placeholder="Dòng đầu ghi tên vị trí, bên dưới là mô tả công việc..."
    )

    job3 = st.text_area(
        "Vị trí 3 · Không bắt buộc",
        height=110,
        placeholder="Nhập thêm vị trí nếu cần..."
    )

    if st.button(
        "📊 So sánh các vị trí",
        use_container_width=True,
        key="run_comparison"
    ):
        if not uploaded_cv:
            st.error("Vui lòng tải CV.")

        elif not job1.strip() or not job2.strip():
            st.error("Vui lòng nhập ít nhất hai mô tả công việc.")

        else:
            try:
                with st.spinner("Đang đọc CV và so sánh..."):
                    cv_text = extract_cv_text(uploaded_cv)

                    if len(cv_text) < 30:
                        st.error(
                            "Không trích xuất được đủ văn bản. "
                            "Hãy thử file rõ nét hơn."
                        )
                    else:
                        jobs = [job1.strip(), job2.strip()]

                        if job3.strip():
                            jobs.append(job3.strip())

                        rows = []

                        for index, job in enumerate(jobs, start=1):
                            score = calculate_similarity(cv_text, job)

                            first_line = job.splitlines()[0].strip()
                            title = first_line[:80] or f"Vị trí {index}"

                            rows.append({
                                "Vị trí": title,
                                "Điểm tương đồng (%)": round(score, 1),
                                "Đánh giá": get_match_level(score)
                            })

                        comparison_df = pd.DataFrame(rows).sort_values(
                            "Điểm tương đồng (%)",
                            ascending=False
                        )

                        st.session_state.compare_results = comparison_df

            except Exception as e:
                st.error(f"Không thể so sánh các vị trí: {e}")

    comparison_df = st.session_state.compare_results

    if comparison_df is not None:
        st.subheader("Kết quả so sánh")

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            comparison_df,
            x="Vị trí",
            y="Điểm tương đồng (%)",
            text="Điểm tương đồng (%)",
            range_y=[0, 100],
            title="Mức độ tương đồng theo vị trí"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#172B4D")
        )

        st.plotly_chart(fig, use_container_width=True)

        best = comparison_df.iloc[0]

        st.success(
            f"Điểm tương đồng cao nhất trong lần so sánh này: "
            f"{best['Vị trí']} — {best['Điểm tương đồng (%)']:.1f}%."
        )

        st.caption(
            "Đây là kết quả so sánh văn bản, không phải dự đoán khả năng "
            "được tuyển dụng."
        )

        st.download_button(
            "⬇️ Tải kết quả CSV",
            data=comparison_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="CVisionAI_so_sanh.csv",
            mime="text/csv"
        )


# =========================================================
# 12. TRANG GỢI Ý VIẾT LẠI CV
# =========================================================

elif page == "Gợi ý viết lại CV":
    show_hero(
        "Gợi ý cải thiện CV",
        "Rà soát cách trình bày kinh nghiệm, dự án và kỹ năng "
        "để hồ sơ rõ ràng và liên quan hơn đến vị trí ứng tuyển."
    )

    result = st.session_state.analysis_result

    if not result:
        st.info(
            "Hãy thực hiện phân tích CV trước để xem các gợi ý "
            "dựa trên kết quả hiện tại."
        )

    else:
        st.write(f"**CV:** {result['filename']}")
        st.write(f"**Vị trí:** {result['job_title']}")

        st.markdown("### 1. Các kỹ năng cần xem xét")

        if result["missing_skills"]:
            for skill in result["missing_skills"]:
                st.markdown(
                    f"- Kiểm tra xem bạn có kinh nghiệm liên quan đến "
                    f"**{skill}** hay không. Nếu có, hãy thể hiện bằng "
                    f"ví dụ thực tế."
                )
        else:
            st.write(
                "Không phát hiện từ khóa thiếu trong danh sách kỹ năng "
                "được thiết lập. Bạn vẫn nên rà soát lại mô tả công việc."
            )

        st.markdown("### 2. Công thức mô tả kinh nghiệm")

        st.markdown("""
        - **Hành động:** Bạn đã thực hiện công việc gì?
        - **Phạm vi:** Công việc thuộc dự án hoặc nhiệm vụ nào?
        - **Kết quả:** Đạt được kết quả gì có thể kiểm chứng?

        Ví dụ:

        - Viết chung chung: “Hỗ trợ hoạt động truyền thông.”
        - Cụ thể hơn: “Hỗ trợ xây dựng nội dung cho chiến dịch truyền thông
          trên mạng xã hội, theo dõi tương tác và tổng hợp kết quả.”
        
        Chỉ sử dụng ví dụ phù hợp với trải nghiệm thật của bạn.
        """)

        st.markdown("### 3. Chỉnh sửa một đoạn CV")

    original_text = st.text_area(
        "Dán nội dung CV muốn cải thiện",
        height=150,
        placeholder="Dán phần giới thiệu, kinh nghiệm hoặc dự án..."
    )

    style = st.selectbox(
        "Phong cách",
        [
            "Chuyên nghiệp, ngắn gọn",
            "Nhấn mạnh kết quả",
            "Phù hợp sinh viên / thực tập sinh"
        ]
    )

    if st.button(
        "✨ Tạo gợi ý chỉnh sửa",
        use_container_width=True,
        key="create_rewrite"
    ):
        if not original_text.strip():
            st.warning("Vui lòng nhập đoạn CV cần cải thiện.")
        else:
            st.session_state.rewrite_output = f"""
PHONG CÁCH: {style}

NỘI DUNG BAN ĐẦU:
{original_text.strip()}

GỢI Ý CHỈNH SỬA:

1. Dùng động từ hành động cụ thể.
2. Nêu rõ nhiệm vụ, phạm vi công việc hoặc dự án.
3. Bổ sung kết quả có thể kiểm chứng nếu có.
4. Ưu tiên thông tin liên quan đến vị trí ứng tuyển.
5. Kiểm tra lỗi chính tả và cách trình bày.

LƯU Ý:
Đây là gợi ý theo quy tắc, chưa phải bản viết lại tự động bằng
mô hình ngôn ngữ tạo sinh. Hãy điều chỉnh để phản ánh chính xác
kinh nghiệm và năng lực thực tế của bạn.
"""

    if st.session_state.rewrite_output:
        st.text_area(
            "Nội dung gợi ý",
            value=st.session_state.rewrite_output,
            height=260
        )

        st.download_button(
            "⬇️ Tải gợi ý chỉnh sửa",
            data=st.session_state.rewrite_output.encode("utf-8"),
            file_name="CVisionAI_goi_y_CV.txt",
            mime="text/plain"
        )


# =========================================================
# 13. TRANG LUYỆN PHỎNG VẤN
# =========================================================

elif page == "Luyện phỏng vấn":
    show_hero(
        "Luyện phỏng vấn",
        "Chuẩn bị câu trả lời và tự luyện tập trước khi tham gia "
        "phỏng vấn tuyển dụng."
    )

    result = st.session_state.analysis_result

    role = st.text_input(
        "Vị trí ứng tuyển",
        value=result["job_title"] if result else "",
        placeholder="Ví dụ: Marketing Intern"
    )

    interview_type = st.selectbox(
        "Nhóm câu hỏi",
        [
            "Câu hỏi chung",
            "Kỹ năng chuyên môn",
            "Kinh nghiệm và dự án",
            "Tình huống xử lý vấn đề"
        ]
    )

    question_bank = {
        "Câu hỏi chung": [
            "Bạn hãy giới thiệu ngắn gọn về bản thân.",
            "Vì sao bạn quan tâm đến vị trí này?",
            "Điểm mạnh nào giúp bạn phù hợp với công việc?",
            "Bạn muốn phát triển kỹ năng nào trong thời gian tới?",
            "Bạn mong đợi điều gì từ môi trường làm việc?"
        ],
        "Kỹ năng chuyên môn": [
            "Bạn bắt đầu một nhiệm vụ chuyên môn mới như thế nào?",
            "Bạn thường sử dụng công cụ nào để hỗ trợ công việc?",
            "Bạn đánh giá hiệu quả công việc bằng chỉ số nào?",
            "Bạn cập nhật kiến thức chuyên môn bằng cách nào?",
            "Hãy mô tả cách bạn hoàn thành một nhiệm vụ có thời hạn ngắn."
        ],
        "Kinh nghiệm và dự án": [
            "Hãy kể về một dự án bạn từng tham gia.",
            "Vai trò cụ thể của bạn trong dự án là gì?",
            "Khó khăn lớn nhất bạn từng gặp là gì?",
            "Bạn rút ra bài học gì từ dự án đó?",
            "Nếu làm lại, bạn sẽ thay đổi điều gì?"
        ],
        "Tình huống xử lý vấn đề": [
            "Bạn làm gì khi nhận nhiệm vụ chưa rõ yêu cầu?",
            "Bạn xử lý thế nào khi nhiều công việc cùng thời hạn?",
            "Bạn phản hồi ra sao khi nhận góp ý trái chiều?",
            "Bạn làm gì khi kết quả không đạt mục tiêu?",
            "Bạn giải quyết thế nào khi bất đồng với đồng đội?"
        ]
    }

    if st.button(
        "🎤 Tạo bộ câu hỏi",
        use_container_width=True,
        key="generate_interview"
    ):
        st.session_state.interview_questions = question_bank[interview_type]

    if st.session_state.interview_questions:
        st.subheader(
            f"Câu hỏi tham khảo cho: {role or 'Vị trí chưa xác định'}"
        )

        for i, question in enumerate(
            st.session_state.interview_questions,
            start=1
        ):
            with st.expander(f"Câu {i}: {question}"):
                st.text_area(
                    "Câu trả lời của bạn",
                    key=f"interview_answer_{i}",
                    height=100,
                    placeholder="Nhập câu trả lời để tự luyện tập..."
                )

        st.info(
            "Câu hỏi được chọn từ bộ mẫu có sẵn, chưa phải nội dung "
            "được AI tạo sinh tự động."
        )


# =========================================================
# 14. TRANG LỊCH SỬ PHÂN TÍCH
# =========================================================

elif page == "Lịch sử phân tích":
    show_hero(
        "Lịch sử phân tích",
        "Theo dõi kết quả trong phiên hiện tại và tải dữ liệu "
        "để lưu trữ."
    )

    history = st.session_state.history

    if not history:
        st.info(
            "Chưa có lịch sử. Hãy thực hiện phân tích CV trước."
        )

    else:
        df = pd.DataFrame(history)

        st.metric("Tổng số lượt phân tích", len(df))

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Xuất lịch sử CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="CVisionAI_lich_su.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Xóa lịch sử trong phiên"):
            st.session_state.history = []
            st.rerun()

        st.caption(
            "Lịch sử chỉ được lưu trong phiên làm việc hiện tại. "
            "Tải CSV xuống để lưu trữ lâu dài."
        )


# =========================================================
# 15. TRANG HƯỚNG DẪN
# =========================================================

elif page == "Hướng dẫn sử dụng":
    show_hero(
        "Hướng dẫn sử dụng",
        "Các bước sử dụng CVision AI."
    )

    steps = [
        (
            "Bước 1 · Chuẩn bị CV",
            "Chuẩn bị PDF, JPG, JPEG hoặc PNG. Ảnh nên rõ nét, "
            "đủ sáng và không bị nghiêng."
        ),
        (
            "Bước 2 · Nhập mô tả công việc",
            "Dán nhiệm vụ, yêu cầu kỹ năng và tiêu chí tuyển dụng."
        ),
        (
            "Bước 3 · Chạy phân tích",
            "Nhấn nút phân tích và chờ mô hình xử lý. Ảnh và PDF scan "
            "có thể cần thêm thời gian cho OCR."
        ),
        (
            "Bước 4 · Đọc kết quả",
            "Xem điểm tương đồng và danh sách từ khóa kỹ năng."
        ),
        (
            "Bước 5 · Cải thiện CV",
            "Đối chiếu các gợi ý với kinh nghiệm thực tế và chỉ bổ sung "
            "thông tin chính xác."
        )
    ]

    for title, description in steps:
        st.markdown(
            f"""
            <div class="card">
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Lưu ý")

    st.markdown("""
    - PDF scan và ảnh được đọc bằng OCR, nên có thể xảy ra lỗi nhận dạng.
    - Chữ viết tay hoặc bố cục nhiều cột phức tạp có thể làm giảm độ chính xác.
    - Điểm tương đồng không phải xác suất được tuyển dụng.
    - Lịch sử hiện chỉ tồn tại trong phiên làm việc.
    - Mô hình ngữ nghĩa có thể mất thời gian tải ở lần chạy đầu.
    """)


# =========================================================
# 16. TRANG GIỚI THIỆU
# =========================================================

elif page == "Giới thiệu":
    show_hero(
        "Giới thiệu CVision AI",
        "Công cụ hỗ trợ ứng viên rà soát hồ sơ và so sánh CV "
        "với yêu cầu công việc."
    )

    st.markdown("""
    <div class="card">
        <h3>🎯 Mục tiêu</h3>
        <p>
        Hỗ trợ ứng viên kiểm tra nội dung hồ sơ, nhận diện từ khóa kỹ năng
        và tham khảo mức độ tương đồng với mô tả công việc.
        </p>
    </div>

    <div class="card">
        <h3>🧠 Công nghệ</h3>
        <p>
        Sentence Transformers tạo biểu diễn ngữ nghĩa của văn bản.
        Cosine similarity được sử dụng để tính độ tương đồng.
        PyMuPDF trích xuất văn bản từ PDF; Tesseract OCR nhận dạng chữ
        từ ảnh và PDF scan. Streamlit cung cấp giao diện ứng dụng.
        </p>
    </div>

    <div class="card">
        <h3>⚖️ Giới hạn</h3>
        <p>
        Điểm tương đồng không phải xác suất được tuyển dụng.
        Việc phát hiện kỹ năng dựa trên danh sách từ khóa và có thể bỏ sót
        từ đồng nghĩa. Gợi ý viết lại và câu hỏi phỏng vấn hiện sử dụng
        quy tắc và mẫu có sẵn.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Thông tin mô hình")

    st.write(
        "**Mô hình ngữ nghĩa:** "
        "`paraphrase-multilingual-MiniLM-L12-v2`"
    )
    st.write("**Phương pháp:** Cosine similarity")
    st.write("**Đọc PDF:** PyMuPDF")
    st.write("**OCR ảnh:** Tesseract OCR")
    st.write("**Giao diện:** Streamlit")
    st.write("**Ngôn ngữ:** Tiếng Việt")

    st.warning(
        "CVision AI là công cụ hỗ trợ tham khảo, không thay thế "
        "đánh giá của ứng viên hoặc nhà tuyển dụng."
    )


# =========================================================
# 17. CHÂN TRANG
# =========================================================

st.markdown("---")

st.markdown("""
<div style="
    text-align:center;
    color:#64748B;
    font-size:12px;
    padding:8px;
">
    CVision AI · Career Intelligence Platform<br>
    Công cụ hỗ trợ phân tích CV và định hướng nghề nghiệp
</div>
""", unsafe_allow_html=True)

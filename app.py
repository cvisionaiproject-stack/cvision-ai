
import streamlit as st
import fitz
import re
import io
import pandas as pd
import plotly.express as px
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. CẤU HÌNH TRANG
# =========================================================

st.set_page_config(
    page_title="CVision AI | Phân tích hồ sơ",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. GIAO DIỆN: NAVY - XANH DƯƠNG - TRẮNG
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #102A43;
    --navy-light: #173F75;
    --blue: #2563EB;
    --blue-light: #60A5FA;
    --blue-pale: #DBEAFE;
    --page: #F3F7FC;
    --white: #FFFFFF;
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

[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 14px;
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
# 3. KHỞI TẠO DỮ LIỆU PHIÊN LÀM VIỆC
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "last_cv_text" not in st.session_state:
    st.session_state.last_cv_text = ""

if "last_job_text" not in st.session_state:
    st.session_state.last_job_text = ""

if "last_filename" not in st.session_state:
    st.session_state.last_filename = ""


# =========================================================
# 4. MÔ HÌNH AI
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def calculate_similarity(cv_text, job_text):
    """Tính độ tương đồng ngữ nghĩa giữa CV và mô tả công việc."""
    model = load_model()

    embeddings = model.encode(
        [cv_text, job_text],
        convert_to_numpy=True
    )

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return max(0.0, min(1.0, float(score))) * 100


# =========================================================
# 5. ĐỌC NỘI DUNG CV PDF
# =========================================================

def extract_pdf_text(uploaded_file):
    """Trích xuất văn bản từ file PDF có lớp văn bản."""
    pdf_bytes = uploaded_file.getvalue()
    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    for page in document:
        pages.append(page.get_text("text"))

    document.close()
    return "\n".join(pages).strip()


# =========================================================
# 6. PHÂN TÍCH TỪ KHÓA
# =========================================================

SKILL_LIST = [
    "marketing", "digital marketing", "seo", "sem",
    "content marketing", "social media", "facebook ads",
    "google ads", "email marketing", "branding",
    "market research", "data analysis", "excel",
    "power bi", "tableau", "python", "sql",
    "communication", "teamwork", "leadership",
    "problem solving", "project management",
    "customer service", "sales", "copywriting",
    "graphic design", "canva", "photoshop",
    "english", "tiếng anh", "phân tích dữ liệu",
    "giao tiếp", "làm việc nhóm", "quản lý dự án",
    "nghiên cứu thị trường", "sáng tạo nội dung",
    "quảng cáo", "thương hiệu", "thực tập",
    "kinh doanh", "bán hàng", "thuyết trình",
    "tin học văn phòng", "word", "powerpoint"
]


def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def find_skills(text):
    """Tìm các kỹ năng trong văn bản bằng danh sách từ khóa."""
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
# 7. THANH ĐIỀU HƯỚNG
# =========================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 18px 0;">
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
            "Giới thiệu",
        ],
        label_visibility="visible"
    )

    st.divider()
    st.caption("CVISION AI · CAREER PLATFORM")


# =========================================================
# 8. HÀM HIỂN THỊ PHẦN ĐẦU TRANG
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


# =========================================================
# 9. TRANG TỔNG QUAN
# =========================================================

if page == "Tổng quan":
    show_hero(
        "CVision AI",
        "Nền tảng hỗ trợ phân tích hồ sơ ứng tuyển, đánh giá mức độ phù hợp "
        "với mô tả công việc và định hướng cải thiện CV bằng AI."
    )

    st.subheader("Chào mừng đến với CVision AI")
    st.write(
        "Bắt đầu bằng cách tải CV của bạn lên, nhập mô tả công việc "
        "và xem những điểm cần cải thiện trước khi ứng tuyển."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>📄 Phân tích CV</h3>
            <p>Đọc nội dung CV và so sánh ngữ nghĩa với yêu cầu tuyển dụng.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>🎯 So sánh vị trí</h3>
            <p>Đối chiếu một CV với nhiều mô tả công việc để tham khảo lựa chọn.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h3>✨ Cải thiện hồ sơ</h3>
            <p>Nhận gợi ý về kỹ năng, nội dung CV và câu hỏi luyện phỏng vấn.</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Tổng quan hoạt động trong phiên hiện tại")

    history = st.session_state.history
    total = len(history)

    if total:
        avg_score = sum(x["Điểm tương đồng (%)"] for x in history) / total
        best_score = max(x["Điểm tương đồng (%)"] for x in history)
    else:
        avg_score = 0
        best_score = 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Lượt phân tích", total)
    c2.metric("Điểm tương đồng trung bình", f"{avg_score:.1f}%")
    c3.metric("Điểm tương đồng cao nhất", f"{best_score:.1f}%")

    st.info(
        "Điểm tương đồng là chỉ số tham khảo dựa trên nội dung văn bản, "
        "không phải xác suất được tuyển dụng."
    )


# =========================================================
# 10. TRANG PHÂN TÍCH CV
# =========================================================

elif page == "Phân tích CV":
    show_hero(
        "Phân tích CV",
        "Tải hồ sơ PDF và nhập mô tả công việc để đánh giá mức độ tương đồng, "
        "từ khóa kỹ năng và các nội dung nên bổ sung."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("01 · Hồ sơ ứng viên")
        uploaded_cv = st.file_uploader(
            "Tải CV định dạng PDF",
            type=["pdf"],
            key="analysis_cv"
        )

        if uploaded_cv:
            st.caption(f"File đã chọn: {uploaded_cv.name}")

    with col2:
        st.subheader("02 · Mô tả công việc")
        job_title = st.text_input(
            "Tên vị trí ứng tuyển",
            placeholder="Ví dụ: Marketing Intern",
            key="analysis_job_title"
        )

        job_description = st.text_area(
            "Dán mô tả công việc (Job Description)",
            placeholder=(
                "Nhập nhiệm vụ, yêu cầu kỹ năng, kinh nghiệm "
                "và tiêu chí tuyển dụng..."
            ),
            height=220,
            key="analysis_job_description"
        )

    if st.button("🚀 Phân tích mức độ phù hợp", use_container_width=True):
        if not uploaded_cv:
            st.error("Vui lòng tải CV PDF trước khi phân tích.")
        elif not job_description.strip():
            st.error("Vui lòng nhập mô tả công việc.")
        else:
            try:
                with st.spinner("Đang đọc CV và chạy mô hình AI..."):
                    cv_text = extract_pdf_text(uploaded_cv)

                    if len(cv_text.strip()) < 30:
                        st.error(
                            "Không trích xuất được đủ văn bản từ PDF. "
                            "Nếu đây là PDF scan hoặc ảnh chụp, hệ thống "
                            "hiện chưa hỗ trợ OCR. Hãy thử PDF có thể chọn "
                            "và sao chép văn bản."
                        )
                        st.stop()

                    score = calculate_similarity(
                        cv_text,
                        job_description
                    )

                    cv_skills = find_skills(cv_text)
                    job_skills = find_skills(job_description)

                    matched_skills = [
                        s for s in job_skills if s in cv_skills
                    ]
                    missing_skills = [
                        s for s in job_skills if s not in cv_skills
                    ]

                    result = {
                        "filename": uploaded_cv.name,
                        "job_title": job_title.strip() or "Chưa đặt tên vị trí",
                        "score": score,
                        "cv_text": cv_text,
                        "job_text": job_description,
                        "cv_skills": cv_skills,
                        "job_skills": job_skills,
                        "matched_skills": matched_skills,
                        "missing_skills": missing_skills
                    }

                    st.session_state.analysis_result = result
                    st.session_state.last_cv_text = cv_text
                    st.session_state.last_job_text = job_description
                    st.session_state.last_filename = uploaded_cv.name

                    st.session_state.history.append({
                        "Thời gian": datetime.now().strftime("%d/%m/%Y %H:%M"),
                        "Tên CV": uploaded_cv.name,
                        "Vị trí": result["job_title"],
                        "Điểm tương đồng (%)": round(score, 1),
                        "Kỹ năng trùng khớp": len(matched_skills),
                        "Kỹ năng cần xem xét": len(missing_skills)
                    })

            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi phân tích: {e}")

    result = st.session_state.analysis_result

    if result:
        st.divider()
        st.subheader("Kết quả phân tích")

        m1, m2, m3 = st.columns(3)
        m1.metric("Điểm tương đồng", f"{result['score']:.1f}%")
        m2.metric("Kỹ năng trùng khớp", len(result["matched_skills"]))
        m3.metric("Kỹ năng cần xem xét", len(result["missing_skills"]))

        st.progress(int(result["score"]) / 100)
        st.markdown(f"**Đánh giá:** {get_match_level(result['score'])}")

        st.caption(
            "Điểm số được tính từ độ tương đồng ngữ nghĩa giữa CV và mô tả "
            "công việc. Đây không phải điểm ATS chính thức hoặc xác suất "
            "được nhà tuyển dụng lựa chọn."
        )

        left, right = st.columns(2)

        with left:
            st.markdown("### ✅ Từ khóa kỹ năng tìm thấy trong cả hai")
            if result["matched_skills"]:
                st.markdown(
                    " ".join(
                        f'<span class="good">{s}</span>'
                        for s in result["matched_skills"]
                    ),
                    unsafe_allow_html=True
                )
            else:
                st.write(
                    "Chưa phát hiện kỹ năng trùng khớp từ danh sách từ khóa "
                    "được thiết lập."
                )

        with right:
            st.markdown("### 🔎 Kỹ năng trong JD chưa thấy trong CV")
            if result["missing_skills"]:
                st.markdown(
                    " ".join(
                        f'<span class="missing">{s}</span>'
                        for s in result["missing_skills"]
                    ),
                    unsafe_allow_html=True
                )
                st.caption(
                    "Hãy kiểm tra từng kỹ năng và chỉ bổ sung nếu bạn thực sự có."
                )
            else:
                st.write(
                    "Không phát hiện từ khóa kỹ năng còn thiếu trong danh sách "
                    "từ khóa được thiết lập."
                )

        if result["job_skills"]:
            chart_df = pd.DataFrame({
                "Nhóm": ["Đã tìm thấy", "Chưa thấy trong CV"],
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
                "Kiểm tra xem CV có nhắm đúng vị trí đang ứng tuyển hay không."
            )

        if result["missing_skills"]:
            suggestions.append(
                "Đối chiếu các kỹ năng còn thiếu với kinh nghiệm thực tế của bạn."
            )

        if not re.search(
            r"\b\d+%|\b\d+\s*(năm|tháng|dự án|khách hàng|người)\b",
            result["cv_text"].lower()
        ):
            suggestions.append(
                "Cân nhắc thể hiện kết quả bằng số liệu cụ thể nếu có căn cứ, "
                "chẳng hạn số dự án hoặc kết quả công việc."
            )

        suggestions.append(
            "Ưu tiên các kinh nghiệm và dự án liên quan trực tiếp đến vị trí."
        )
        suggestions.append(
            "Không thêm kỹ năng hoặc thành tích mà bạn không thực sự có."
        )

        for item in suggestions:
            st.markdown(f"- {item}")

        report = f"""BÁO CÁO PHÂN TÍCH CVISION AI

Tên CV: {result['filename']}
Vị trí ứng tuyển: {result['job_title']}
Điểm tương đồng: {result['score']:.1f}%
Đánh giá: {get_match_level(result['score'])}

KỸ NĂNG TRÙNG KHỚP
{', '.join(result['matched_skills']) or 'Chưa phát hiện'}

KỸ NĂNG TRONG JD CHƯA THẤY TRONG CV
{', '.join(result['missing_skills']) or 'Chưa phát hiện'}

LƯU Ý
Điểm tương đồng là chỉ số tham khảo, không phải xác suất được tuyển dụng.
Kết quả từ khóa phụ thuộc vào danh sách kỹ năng được thiết lập.
"""
        st.download_button(
            "⬇️ Tải báo cáo phân tích (.txt)",
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
        "Đối chiếu cùng một CV với nhiều mô tả công việc để tham khảo "
        "mức độ tương đồng giữa hồ sơ và từng vị trí."
    )

    uploaded_cv = st.file_uploader(
        "Tải CV PDF để so sánh",
        type=["pdf"],
        key="compare_cv"
    )

    st.write("Nhập ít nhất hai mô tả công việc để bắt đầu so sánh.")

    job1 = st.text_area(
        "Vị trí 1 · Tên và mô tả công việc",
        height=140,
        placeholder="Ví dụ: Marketing Intern\nMô tả công việc..."
    )

    job2 = st.text_area(
        "Vị trí 2 · Tên và mô tả công việc",
        height=140,
        placeholder="Ví dụ: Sales Intern\nMô tả công việc..."
    )

    job3 = st.text_area(
        "Vị trí 3 · Tên và mô tả công việc (không bắt buộc)",
        height=120,
        placeholder="Nhập thêm nếu muốn so sánh vị trí thứ ba..."
    )

    if st.button("📊 So sánh các vị trí", use_container_width=True):
        if not uploaded_cv:
            st.error("Vui lòng tải CV PDF.")
        elif not job1.strip() or not job2.strip():
            st.error("Vui lòng nhập ít nhất hai mô tả công việc.")
        else:
            try:
                cv_text = extract_pdf_text(uploaded_cv)

                if len(cv_text) < 30:
                    st.error(
                        "Không trích xuất được đủ văn bản từ PDF. "
                        "Hãy thử file PDF có thể chọn và sao chép văn bản."
                    )
                    st.stop()

                jobs = [job1.strip(), job2.strip()]
                if job3.strip():
                    jobs.append(job3.strip())

                rows = []
                with st.spinner("Đang so sánh CV với các vị trí..."):
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

                st.session_state["compare_results"] = comparison_df

            except Exception as e:
                st.error(f"Không thể hoàn thành so sánh: {e}")

    if "compare_results" in st.session_state:
        comparison_df = st.session_state["compare_results"]

        st.subheader("Kết quả so sánh")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

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
            f"Vị trí có điểm tương đồng cao nhất trong lần so sánh này: "
            f"{best['Vị trí']} ({best['Điểm tương đồng (%)']:.1f}%)."
        )

        st.caption(
            "Đây chỉ là kết quả so sánh văn bản. Hãy cân nhắc thêm sở thích, "
            "điều kiện thực tế và yêu cầu tuyển dụng trước khi quyết định."
        )

        st.download_button(
            "⬇️ Tải kết quả so sánh CSV",
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
        "Xem các gợi ý cấu trúc và cách diễn đạt để làm hồ sơ rõ ràng, "
        "dễ đọc và liên quan hơn đến vị trí ứng tuyển."
    )

    result = st.session_state.analysis_result

    if not result:
        st.info(
            "Hãy vào mục Phân tích CV và hoàn thành ít nhất một lần phân tích "
            "để sử dụng các gợi ý dựa trên CV và mô tả công việc."
        )
    else:
        st.subheader(f"CV đang xem: {result['filename']}")
        st.write(f"**Vị trí:** {result['job_title']}")

        st.markdown("### 1. Kỹ năng nên đối chiếu")
        if result["missing_skills"]:
            for skill in result["missing_skills"]:
                st.markdown(
                    f"- Kiểm tra xem bạn có kinh nghiệm hoặc năng lực "
                    f"liên quan đến **{skill}** hay không. "
                    f"Nếu có, hãy thể hiện bằng ví dụ thực tế."
                )
        else:
            st.write(
                "Không phát hiện từ khóa kỹ năng thiếu trong danh sách từ khóa "
                "được thiết lập. Bạn vẫn nên rà soát lại mô tả công việc."
            )

        st.markdown("### 2. Gợi ý cách mô tả kinh nghiệm")
        st.markdown("""
        **Công thức tham khảo:**

        - Hành động: Bạn đã làm gì?
        - Phạm vi: Bạn thực hiện trong dự án hoặc công việc nào?
        - Kết quả: Đạt được kết quả gì có thể kiểm chứng?

        Ví dụ minh họa:

        - Cách viết chung chung: “Hỗ trợ hoạt động truyền thông.”
        - Cách viết cụ thể hơn: “Hỗ trợ xây dựng nội dung cho chiến dịch truyền
          thông trên mạng xã hội; theo dõi tương tác và tổng hợp kết quả.”

        Chỉ sử dụng ví dụ phù hợp với trải nghiệm thật của bạn.
        """)

        st.markdown("### 3. Viết lại một đoạn CV")
        original_text = st.text_area(
            "Dán đoạn CV bạn muốn cải thiện",
            value="",
            height=160,
            placeholder="Dán phần giới thiệu bản thân, kinh nghiệm hoặc dự án..."
        )

        style = st.selectbox(
            "Phong cách diễn đạt",
            [
                "Chuyên nghiệp, ngắn gọn",
                "Nhấn mạnh kết quả",
                "Phù hợp sinh viên / thực tập sinh"
            ]
        )

        if st.button("✨ Tạo bản gợi ý", use_container_width=True):
            if not original_text.strip():
                st.warning("Vui lòng nhập đoạn CV cần cải thiện.")
            else:
                st.session_state["rewrite_output"] = (
                    f"PHONG CÁCH: {style}\n\n"
                    "BẢN GỢI Ý CHỈNH SỬA\n\n"
                    f"{original_text.strip()}\n\n"
                    "Gợi ý chỉnh sửa:\n"
                    "• Dùng động từ hành động rõ ràng.\n"
                    "• Nêu nhiệm vụ hoặc phạm vi công việc cụ thể.\n"
                    "• Bổ sung kết quả có thể kiểm chứng nếu có.\n"
                    "• Đối chiếu với mô tả công việc để ưu tiên thông tin liên quan.\n\n"
                    "Lưu ý: Đây là khung gợi ý theo quy tắc, chưa phải bản viết lại "
                    "tự động bằng mô hình ngôn ngữ sinh nội dung. Hãy tự điều chỉnh "
                    "để phản ánh chính xác kinh nghiệm thật."
                )

        if st.session_state.get("rewrite_output"):
            st.text_area(
                "Nội dung gợi ý",
                value=st.session_state["rewrite_output"],
                height=260
            )
            st.download_button(
                "⬇️ Tải gợi ý (.txt)",
                data=st.session_state["rewrite_output"].encode("utf-8"),
                file_name="CVisionAI_goi_y_CV.txt",
                mime="text/plain"
            )


# =========================================================
# 13. TRANG LUYỆN PHỎNG VẤN
# =========================================================

elif page == "Luyện phỏng vấn":
    show_hero(
        "Luyện phỏng vấn",
        "Chuẩn bị trước các câu hỏi thường gặp và xây dựng câu trả lời "
        "dựa trên trải nghiệm thực tế của bạn."
    )

    result = st.session_state.analysis_result

    role = st.text_input(
        "Vị trí muốn luyện phỏng vấn",
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
            "Bạn sẽ bắt đầu thực hiện một nhiệm vụ chuyên môn mới như thế nào?",
            "Bạn thường dùng công cụ nào để hỗ trợ công việc?",
            "Bạn đánh giá hiệu quả công việc bằng những chỉ số nào?",
            "Bạn sẽ cập nhật kiến thức chuyên môn bằng cách nào?",
            "Hãy mô tả cách bạn giải quyết một nhiệm vụ có thời hạn ngắn."
        ],
        "Kinh nghiệm và dự án": [
            "Hãy kể về một dự án bạn từng tham gia.",
            "Vai trò cụ thể của bạn trong dự án là gì?",
            "Khó khăn lớn nhất bạn gặp phải là gì?",
            "Bạn đã rút ra bài học gì từ dự án đó?",
            "Nếu làm lại, bạn sẽ thay đổi điều gì?"
        ],
        "Tình huống xử lý vấn đề": [
            "Bạn sẽ làm gì khi nhận nhiệm vụ chưa rõ yêu cầu?",
            "Bạn xử lý thế nào khi có nhiều công việc cùng thời hạn?",
            "Bạn sẽ phản hồi ra sao khi nhận góp ý trái chiều?",
            "Bạn làm gì khi kết quả công việc không đạt mục tiêu?",
            "Bạn xử lý thế nào khi bất đồng ý kiến với đồng đội?"
        ]
    }

    if st.button("🎤 Tạo bộ câu hỏi", use_container_width=True):
        st.session_state["interview_questions"] = question_bank[interview_type]

    questions = st.session_state.get("interview_questions", [])

    if questions:
        st.subheader(f"Câu hỏi tham khảo cho vị trí: {role or 'Chưa xác định'}")

        for i, question in enumerate(questions, start=1):
            with st.expander(f"Câu {i}: {question}"):
                st.text_area(
                    "Câu trả lời của bạn",
                    key=f"interview_answer_{i}",
                    height=100,
                    placeholder="Nhập câu trả lời để tự luyện tập..."
                )

        st.info(
            "Bộ câu hỏi hiện được tạo từ mẫu quy tắc có sẵn, chưa phải "
            "câu hỏi sinh tự động từ một mô hình AI tạo sinh."
        )


# =========================================================
# 14. TRANG LỊCH SỬ PHÂN TÍCH
# =========================================================

elif page == "Lịch sử phân tích":
    show_hero(
        "Lịch sử phân tích",
        "Theo dõi kết quả phân tích trong phiên làm việc hiện tại "
        "và xuất dữ liệu để lưu trữ."
    )

    history = st.session_state.history

    if not history:
        st.info(
            "Chưa có lịch sử. Hãy thực hiện phân tích CV để xem kết quả tại đây."
        )
    else:
        df = pd.DataFrame(history)

        st.metric("Tổng số lượt phân tích", len(df))
        st.dataframe(df, use_container_width=True, hide_index=True)

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
            "Tải CSV xuống nếu bạn muốn lưu lại lâu dài."
        )


# =========================================================
# 15. TRANG HƯỚNG DẪN SỬ DỤNG
# =========================================================

elif page == "Hướng dẫn sử dụng":
    show_hero(
        "Hướng dẫn sử dụng",
        "Thực hiện các bước dưới đây để phân tích hồ sơ ứng tuyển."
    )

    steps = [
        (
            "Bước 1 · Chuẩn bị CV",
            "Sử dụng file PDF có thể chọn và sao chép văn bản. "
            "PDF scan hoặc ảnh chụp chưa được hỗ trợ OCR."
        ),
        (
            "Bước 2 · Nhập mô tả công việc",
            "Sao chép nhiệm vụ, yêu cầu kỹ năng và tiêu chí tuyển dụng "
            "từ tin tuyển dụng."
        ),
        (
            "Bước 3 · Chạy phân tích",
            "Nhấn nút phân tích và chờ mô hình xử lý nội dung."
        ),
        (
            "Bước 4 · Đọc kết quả",
            "Xem điểm tương đồng, từ khóa kỹ năng tìm thấy và các kỹ năng "
            "trong mô tả công việc chưa được phát hiện trong CV."
        ),
        (
            "Bước 5 · Cải thiện hồ sơ",
            "Dùng gợi ý như danh sách kiểm tra. Chỉ đưa vào CV thông tin "
            "đúng với kinh nghiệm và năng lực của bạn."
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

    st.subheader("Lưu ý về dữ liệu và kết quả")
    st.markdown("""
    - CVision AI không đưa ra quyết định tuyển dụng.
    - Điểm tương đồng không đồng nghĩa với xác suất được nhận việc.
    - Kết quả đối chiếu từ khóa phụ thuộc vào danh sách từ khóa được cài đặt.
    - Mô hình AI cần tải về khi chạy lần đầu, nên lần đầu có thể mất thời gian.
    - Không tải lên thông tin cá nhân nhạy cảm nếu bạn không cần thiết phải phân tích.
    """)


# =========================================================
# 16. TRANG GIỚI THIỆU
# =========================================================

elif page == "Giới thiệu":
    show_hero(
        "Giới thiệu CVision AI",
        "Công cụ hỗ trợ người tìm việc hiểu rõ hơn mức độ tương đồng "
        "giữa hồ sơ cá nhân và yêu cầu công việc."
    )

    st.markdown("""
    <div class="card">
        <h3>🎯 Mục tiêu</h3>
        <p>
        Hỗ trợ ứng viên rà soát CV một cách có hệ thống, xác định các từ khóa
        kỹ năng cần xem xét và tham khảo mức độ tương đồng với mô tả tuyển dụng.
        </p>
    </div>

    <div class="card">
        <h3>🧠 Công nghệ</h3>
        <p>
        Sentence Transformers tạo biểu diễn ngữ nghĩa cho văn bản.
        Cosine similarity được sử dụng để tính độ tương đồng giữa CV
        và mô tả công việc. PyMuPDF trích xuất văn bản từ PDF.
        Streamlit cung cấp giao diện ứng dụng.
        </p>
    </div>

    <div class="card">
        <h3>⚖️ Giới hạn</h3>
        <p>
        Điểm tương đồng không phải xác suất được tuyển dụng. Việc phát hiện
        kỹ năng sử dụng danh sách từ khóa nên có thể bỏ sót từ đồng nghĩa
        hoặc kỹ năng được diễn đạt theo cách khác. Gợi ý viết lại và câu hỏi
        phỏng vấn hiện dựa trên các quy tắc và mẫu có sẵn.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Thông tin mô hình")
    st.write("**Mô hình:** `paraphrase-multilingual-MiniLM-L12-v2`")
    st.write("**Phương pháp so sánh:** Cosine similarity")
    st.write("**Trích xuất PDF:** PyMuPDF")
    st.write("**Giao diện:** Streamlit")
    st.write("**Ngôn ngữ giao diện:** Tiếng Việt")

    st.warning(
        "Đây là công cụ hỗ trợ tham khảo. Ứng viên cần tự kiểm tra tính chính "
        "xác của kết quả và quyết định nội dung đưa vào hồ sơ."
    )


# =========================================================
# 17. CHÂN TRANG
# =========================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#64748B;font-size:12px;padding:8px;">
        CVision AI · Career Intelligence Platform<br>
        Công cụ hỗ trợ phân tích CV và định hướng nghề nghiệp
    </div>
    """,
    unsafe_allow_html=True
)

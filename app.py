
import re
import io

import streamlit as st
import fitz
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. CẤU HÌNH WEBSITE
# =========================
st.set_page_config(
    page_title="CVision AI | Smart CV Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: #F4F7FB;
}
[data-testid="stSidebar"] {
    background: #101D35;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
.hero {
    background: linear-gradient(120deg, #101D35, #145B66);
    padding: 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    margin-bottom: 6px;
}
.hero p {
    color: #D7F5F0;
    margin-bottom: 0;
}
.metric-card {
    background: white;
    border: 1px solid #E0E7EF;
    padding: 18px;
    border-radius: 12px;
    min-height: 110px;
}
.metric-label {
    color: #64748B;
    font-size: 14px;
}
.metric-value {
    color: #10233F;
    font-size: 25px;
    font-weight: 700;
    margin-top: 8px;
}
.section-title {
    color: #10233F;
    font-size: 21px;
    font-weight: 700;
    margin: 18px 0 10px;
}
.stButton > button {
    background: #0F9D91;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 600;
    min-height: 45px;
}
.stButton > button:hover {
    background: #0B8178;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# =========================
# 2. BỘ KỸ NĂNG THAM KHẢO
# =========================
SKILLS = {
    "Python": ["python"],
    "SQL": ["sql", "mysql", "postgresql", "sqlite"],
    "Excel": ["excel", "spreadsheet"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Machine Learning": ["machine learning", "học máy"],
    "Deep Learning": ["deep learning", "học sâu"],
    "Data Analysis": [
        "data analysis", "data analytics",
        "phân tích dữ liệu"
    ],
    "Statistics": ["statistics", "thống kê"],
    "Communication": [
        "communication", "giao tiếp",
        "communicate effectively"
    ],
    "Teamwork": [
        "teamwork", "team work",
        "làm việc nhóm"
    ],
    "Problem Solving": [
        "problem solving", "giải quyết vấn đề"
    ],
    "Project Management": [
        "project management", "quản lý dự án"
    ],
    "Marketing": ["marketing", "tiếp thị"],
    "SEO": ["seo", "search engine optimization"],
    "Content Writing": [
        "content writing", "copywriting",
        "viết nội dung"
    ],
    "Python Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Data Visualization": [
        "data visualization", "trực quan hóa dữ liệu"
    ],
    "Artificial Intelligence": [
        "artificial intelligence", "trí tuệ nhân tạo"
    ],
    "Leadership": ["leadership", "lãnh đạo"],
    "English": [
        "english", "tiếng anh",
        "english proficiency"
    ],
    "Customer Service": [
        "customer service", "chăm sóc khách hàng"
    ],
    "Sales": ["sales", "bán hàng"],
    "Communication Skills": [
        "interpersonal skills", "kỹ năng giao tiếp"
    ],
}


# =========================
# 3. ĐỌC NỘI DUNG PDF
# =========================
def extract_pdf_text(file_bytes):
    text_parts = []

    with fitz.open(
        stream=file_bytes,
        filetype="pdf"
    ) as pdf:
        for page in pdf:
            text_parts.append(page.get_text())

    return "\n".join(text_parts).strip()


# =========================
# 4. NHẬN DIỆN KỸ NĂNG
# =========================
def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        text.lower()
    ).strip()


def extract_skills(text):
    text = normalize_text(text)
    found = []

    for skill, aliases in SKILLS.items():
        for alias in aliases:
            pattern = (
                r"(?<!\w)"
                + re.escape(alias.lower())
                + r"(?!\w)"
            )

            if re.search(pattern, text):
                found.append(skill)
                break

    return sorted(set(found))


# =========================
# 5. MÔ HÌNH AI CÓ SẴN
# =========================
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )


def analyze_match(cv_text, job_text):
    model = load_model()

    embeddings = model.encode(
        [cv_text, job_text],
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    # Điểm tương đồng ngữ nghĩa, không phải xác suất tuyển dụng
    score = round(max(0, min(100, float(similarity) * 100)))

    cv_skills = extract_skills(cv_text)
    job_skills = extract_skills(job_text)

    matched = sorted(set(cv_skills) & set(job_skills))
    missing = sorted(set(job_skills) - set(cv_skills))

    if job_skills:
        skill_coverage = round(
            len(matched) / len(job_skills) * 100
        )
    else:
        skill_coverage = None

    return {
        "score": score,
        "cv_skills": cv_skills,
        "job_skills": job_skills,
        "matched": matched,
        "missing": missing,
        "skill_coverage": skill_coverage,
    }


# =========================
# 6. SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🎯 CVision AI")
    st.caption("SMART CV ANALYZER")
    st.divider()

    st.markdown("### Phân tích hồ sơ")
    st.write("1. Tải CV PDF")
    st.write("2. Nhập mô tả công việc")
    st.write("3. Xem báo cáo AI")

    st.divider()
    st.caption("Phiên bản MVP | Ứng dụng AI hỗ trợ nghề nghiệp")
    st.caption("CV chỉ được xử lý trong phiên làm việc; ứng dụng không chủ động lưu CV vào cơ sở dữ liệu.")


# =========================
# 7. PHẦN ĐẦU TRANG
# =========================
st.markdown("""
<div class="hero">
    <h1>CVision AI</h1>
    <p>Smart CV Analyzer — Phân tích hồ sơ và đối chiếu yêu cầu công việc bằng AI.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(
        '<div class="section-title">📄 Hồ sơ ứng viên</div>',
        unsafe_allow_html=True
    )

    uploaded_cv = st.file_uploader(
        "Tải CV định dạng PDF",
        type=["pdf"],
        help="Ưu tiên PDF có thể bôi đen và sao chép văn bản."
    )

    if uploaded_cv:
        st.success(f"Đã tải lên: {uploaded_cv.name}")

with right:
    st.markdown(
        '<div class="section-title">💼 Vị trí ứng tuyển</div>',
        unsafe_allow_html=True
    )

    job_title = st.text_input(
        "Tên vị trí",
        placeholder="Ví dụ: Data Analyst Intern"
    )

    job_description = st.text_area(
        "Mô tả công việc (Job Description)",
        height=185,
        placeholder=(
            "Nhập yêu cầu công việc, kỹ năng, "
            "kinh nghiệm và trình độ cần có..."
        )
    )

st.caption(
    "Lưu ý: CV chứa ảnh scan có thể không trích xuất được văn bản. "
    "Bản MVP này chưa tích hợp OCR."
)

analyze_button = st.button(
    "✨ Phân tích CV bằng AI",
    use_container_width=True
)


# =========================
# 8. PHÂN TÍCH VÀ HIỂN THỊ
# =========================
if analyze_button:
    if not uploaded_cv:
        st.error("Vui lòng tải CV PDF trước khi phân tích.")
        st.stop()

    if not job_description.strip():
        st.error("Vui lòng nhập mô tả công việc.")
        st.stop()

    if len(job_description.strip()) < 30:
        st.warning(
            "Mô tả công việc khá ngắn. "
            "Bạn nên nhập thêm yêu cầu để kết quả có ý nghĩa hơn."
        )
        st.stop()

    try:
        with st.spinner("Đang đọc CV và chạy mô hình AI..."):
            cv_text = extract_pdf_text(
                uploaded_cv.getvalue()
            )

            if len(cv_text) < 50:
                st.error(
                    "Không đọc được đủ văn bản từ PDF. "
                    "Hãy dùng PDF có văn bản thay vì ảnh scan."
                )
                st.stop()

            result = analyze_match(
                cv_text,
                job_description
            )

        st.session_state["analysis_result"] = result
        st.session_state["job_title"] = job_title.strip()
        st.session_state["cv_filename"] = uploaded_cv.name

    except Exception as error:
        st.error(
            "Chưa thể phân tích. Có thể mô hình đang tải lần đầu "
            "hoặc xảy ra lỗi kết nối."
        )
        st.code(str(error))
        st.info(
            "Kiểm tra kết nối Internet, thư viện và nhật ký ứng dụng."
        )


# =========================
# 9. BÁO CÁO KẾT QUẢ
# =========================
if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]

    st.divider()
    st.markdown(
        '<div class="section-title">📊 Báo cáo phân tích</div>',
        unsafe_allow_html=True
    )

    st.caption(
        f"CV: {st.session_state['cv_filename']} | "
        f"Vị trí: {st.session_state['job_title'] or 'Chưa đặt tên'}"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Độ tương đồng ngữ nghĩa</div>
                <div class="metric-value">{result['score']}/100</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Kỹ năng phù hợp</div>
                <div class="metric-value">{len(result['matched'])}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        missing_count = len(result["missing"])
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Kỹ năng cần bổ sung</div>
                <div class="metric-value">{missing_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "Điểm tương đồng được tính từ biểu diễn ngữ nghĩa của toàn bộ "
        "CV và mô tả công việc. Điểm này không phải xác suất trúng tuyển "
        "hay đánh giá toàn diện năng lực ứng viên."
    )

    tab1, tab2, tab3 = st.tabs([
        "🎯 Kỹ năng",
        "📈 Biểu đồ",
        "💡 Gợi ý cải thiện"
    ])

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Kỹ năng phù hợp")
            if result["matched"]:
                for skill in result["matched"]:
                    st.success(f"✓ {skill}")
            else:
                st.write(
                    "Chưa phát hiện kỹ năng trùng khớp "
                    "trong bộ từ khóa hiện tại."
                )

        with col_b:
            st.subheader("Kỹ năng còn thiếu")
            if result["missing"]:
                for skill in result["missing"]:
                    st.warning(f"＋ {skill}")
            elif result["job_skills"]:
                st.success(
                    "Đã tìm thấy tất cả kỹ năng trong "
                    "bộ từ khóa mà hệ thống nhận diện."
                )
            else:
                st.write(
                    "Chưa nhận diện được kỹ năng trong mô tả. "
                    "Hãy thử nhập tên kỹ năng rõ hơn."
                )

        st.subheader("Kỹ năng nhận diện trong CV")
        if result["cv_skills"]:
            st.write(", ".join(result["cv_skills"]))
        else:
            st.write("Chưa nhận diện được kỹ năng theo bộ từ khóa.")

    with tab2:
        labels = [
            "Độ tương đồng ngữ nghĩa",
            "Độ bao phủ kỹ năng"
        ]

        values = [result["score"]]

        if result["skill_coverage"] is not None:
            values.append(result["skill_coverage"])
            labels.append("Độ bao phủ kỹ năng")

        # Dùng hai cột để thể hiện riêng hai chỉ số,
        # tránh nhầm lẫn giữa hai cách tính.
        chart_labels = ["Độ tương đồng ngữ nghĩa"]
        chart_values = [result["score"]]

        if result["skill_coverage"] is not None:
            chart_labels.append("Độ bao phủ kỹ năng")
            chart_values.append(result["skill_coverage"])

        fig = go.Figure(
            go.Bar(
                x=chart_values,
                y=chart_labels,
                orientation="h",
                text=[f"{v}%" for v in chart_values],
                textposition="auto",
                marker_color=["#0F9D91", "#5277B8"][
                    :len(chart_values)
                ]
            )
        )

        fig.update_layout(
            xaxis=dict(
                title="Điểm (%)",
                range=[0, 100]
            ),
            yaxis_title="",
            height=260,
            margin=dict(l=10, r=20, t=20, b=20),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Độ bao phủ kỹ năng = số kỹ năng yêu cầu được tìm thấy "
            "trong CV / tổng số kỹ năng nhận diện từ mô tả công việc. "
            "Đây là phép đối chiếu từ bộ từ khóa, không phải mô hình AI."
        )

    with tab3:
        suggestions = []

        if result["missing"]:
            suggestions.append(
                "Kiểm tra lại các kỹ năng còn thiếu và chỉ bổ sung "
                "vào CV nếu bạn thực sự có kiến thức hoặc kinh nghiệm."
            )
            suggestions.append(
                "Nếu đã có kinh nghiệm tương ứng, hãy mô tả bằng "
                "dự án, nhiệm vụ hoặc kết quả cụ thể."
            )

        if result["score"] < 50:
            suggestions.append(
                "Đọc lại mô tả công việc và làm rõ các kinh nghiệm "
                "liên quan trong CV. Điểm thấp không đồng nghĩa "
                "với việc bạn không đủ năng lực."
            )

        suggestions.append(
            "Ưu tiên trình bày CV rõ ràng, có thông tin liên hệ, "
            "học vấn, dự án và kinh nghiệm liên quan."
        )

        suggestions.append(
            "Đối chiếu từng yêu cầu tuyển dụng với bằng chứng cụ thể "
            "trong CV trước khi gửi hồ sơ."
        )

        for i, suggestion in enumerate(suggestions, 1):
            st.write(f"**{i}.** {suggestion}")

    st.divider()
    st.caption(
        "CVision AI là công cụ hỗ trợ tham khảo. "
        "Kết quả có thể bỏ sót kỹ năng, đặc biệt khi cách diễn đạt "
        "khác với bộ từ khóa. Không sử dụng điểm số để tự động "
        "loại ứng viên."
    )

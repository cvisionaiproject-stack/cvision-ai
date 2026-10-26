
import re
import html
import streamlit as st
import fitz
import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="CVision AI | Phân tích CV",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. GIAO DIỆN TIẾNG VIỆT - PHONG CÁCH PASTEL
# =========================================================
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

[data-testid="stSidebar"] h1 {
    color: #7161C5;
}

h1, h2, h3 {
    color: #303653;
    font-weight: 800;
}

p, label, .stMarkdown {
    color: #46516B;
}

div.stButton > button {
    background: linear-gradient(90deg, #8777DF, #6ECDBA);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
    min-height: 45px;
}

div.stButton > button:hover {
    color: white;
    border: 1px solid #8777DF;
    filter: brightness(1.04);
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E9E7F5;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 4px 18px rgba(70, 70, 120, 0.05);
}

div[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #B7ADEB;
    border-radius: 14px;
    padding: 12px;
}

textarea, input {
    border-radius: 10px !important;
}

.cv-card {
    background: white;
    border: 1px solid #E8E5F4;
    border-radius: 18px;
    padding: 22px;
    margin: 8px 0 16px 0;
    box-shadow: 0 5px 20px rgba(60, 60, 100, 0.04);
}

.hero {
    background: linear-gradient(115deg, #EAE5FF, #DDF7F0, #E4EEFF);
    border-radius: 22px;
    padding: 28px;
    margin-bottom: 22px;
}

.hero h1 {
    color: #39345F;
    margin-bottom: 8px;
}

.hero p {
    color: #535B78;
    margin-bottom: 0;
}

.skill-good {
    display: inline-block;
    background: #DDF7E9;
    color: #176B49;
    padding: 7px 11px;
    border-radius: 20px;
    margin: 4px;
    font-size: 13px;
}

.skill-missing {
    display: inline-block;
    background: #FFE9E5;
    color: #A33C37;
    padding: 7px 11px;
    border-radius: 20px;
    margin: 4px;
    font-size: 13px;
}

.small-note {
    color: #737C91;
    font-size: 13px;
}

hr {
    border-color: #E7E5F1;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. TẢI MÔ HÌNH AI
# =========================================================
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


# =========================================================
# 4. HÀM ĐỌC NỘI DUNG TỪ CV PDF
# =========================================================
def extract_pdf_text(uploaded_file):
    try:
        pdf_bytes = uploaded_file.getvalue()
        text_parts = []

        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for page in pdf:
                text_parts.append(page.get_text("text"))

        return "\n".join(text_parts).strip()

    except Exception as error:
        st.error("Không thể đọc tệp PDF. Vui lòng thử một tệp khác.")
        return ""


# =========================================================
# 5. DANH SÁCH KỸ NĂNG PHỔ BIẾN
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
    "AI": ["artificial intelligence", "trí tuệ nhân tạo"],
    "Marketing": ["marketing", "digital marketing", "tiếp thị"],
    "SEO": ["seo", "search engine optimization"],
    "Google Analytics": ["google analytics", "ga4"],
    "Quản lý mạng xã hội": ["social media", "quản lý mạng xã hội"],
    "Content Marketing": ["content marketing", "sáng tạo nội dung"],
    "Giao tiếp": ["communication", "giao tiếp"],
    "Làm việc nhóm": ["teamwork", "team work", "làm việc nhóm"],
    "Quản lý dự án": ["project management", "quản lý dự án"],
    "Quản lý thời gian": ["time management", "quản lý thời gian"],
    "PowerPoint": ["powerpoint"],
    "Word": ["microsoft word"],
    "Quản trị kinh doanh": ["business administration", "quản trị kinh doanh"],
    "Tài chính": ["finance", "financial analysis", "tài chính"],
    "Kế toán": ["accounting", "kế toán"],
    "Thiết kế": ["graphic design", "thiết kế đồ họa"],
    "Figma": ["figma"],
    "Canva": ["canva"],
    "Java": ["java"],
    "JavaScript": ["javascript"],
    "HTML/CSS": ["html", "css"],
    "C++": ["c++"],
    "Git": ["git", "github"],
    "Tiếng Anh": ["english", "tiếng anh", "ielts", "toeic"],
}


def find_skills(text):
    normalized = text.lower()
    found = []

    for skill, keywords in SKILL_GROUPS.items():
        for keyword in keywords:
            pattern = r"(?<![a-z0-9])" + re.escape(keyword.lower()) + r"(?![a-z0-9])"
            if re.search(pattern, normalized):
                found.append(skill)
                break

    return found


# =========================================================
# 6. PHÂN TÍCH CV VỚI MÔ TẢ CÔNG VIỆC
# =========================================================
def analyze_cv(model, cv_text, job_text):
    embeddings = model.encode(
        [cv_text, job_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarity = float(
        cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
    )

    # Chuyển độ tương đồng cosine về thang điểm 0–100.
    # Đây là điểm tham khảo của ứng dụng, không phải xác suất trúng tuyển.
    match_score = max(0, min(100, round(similarity * 100)))

    cv_skills = find_skills(cv_text)
    job_skills = find_skills(job_text)

    matched = [skill for skill in job_skills if skill in cv_skills]
    missing = [skill for skill in job_skills if skill not in cv_skills]

    if not job_skills:
        skill_coverage = None
    else:
        skill_coverage = round(len(matched) / len(job_skills) * 100)

    return {
        "score": match_score,
        "cv_skills": cv_skills,
        "job_skills": job_skills,
        "matched": matched,
        "missing": missing,
        "skill_coverage": skill_coverage
    }


# =========================================================
# 7. THANH MENU BÊN TRÁI
# =========================================================
with st.sidebar:
    st.markdown("# 📄 CVision AI")
    st.caption("Trợ lý phân tích CV bằng trí tuệ nhân tạo")
    st.divider()

    page = st.radio(
        "MENU CHÍNH",
        [
            "Tổng quan",
            "Phân tích CV",
            "Hướng dẫn sử dụng",
            "Giới thiệu"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("**Công cụ hỗ trợ nghề nghiệp**")
    st.caption("Tải CV · So sánh yêu cầu · Gợi ý cải thiện")


# =========================================================
# 8. TRANG TỔNG QUAN
# =========================================================
if page == "Tổng quan":
    st.markdown("""
    <div class="hero">
        <h1>Chào mừng đến với CVision AI ✨</h1>
        <p>
        Khám phá mức độ phù hợp giữa hồ sơ của bạn và vị trí tuyển dụng.
        Nhận diện kỹ năng nổi bật, kỹ năng còn thiếu và gợi ý hoàn thiện CV.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="cv-card">
            <h3>📑 Phân tích hồ sơ</h3>
            <p>Tải CV định dạng PDF để trích xuất và phân tích nội dung.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="cv-card">
            <h3>🎯 Đối chiếu tuyển dụng</h3>
            <p>So sánh nội dung CV với mô tả công việc và xem các kỹ năng liên quan.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("Bắt đầu bằng cách chọn mục **Phân tích CV** ở menu bên trái.")


# =========================================================
# 9. TRANG PHÂN TÍCH CV
# =========================================================
elif page == "Phân tích CV":
    st.markdown("""
    <div class="hero">
        <h1>Phân tích mức độ phù hợp của CV 🔎</h1>
        <p>Tải hồ sơ, nhập mô tả công việc và bắt đầu phân tích.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Bước 1. Tải lên CV")
    uploaded_cv = st.file_uploader(
        "Chọn tệp CV định dạng PDF",
        type=["pdf"],
        help="Tệp PDF cần chứa văn bản có thể chọn và sao chép."
    )

    cv_text = ""
    if uploaded_cv is not None:
        cv_text = extract_pdf_text(uploaded_cv)

        if cv_text:
            st.success("Đã đọc được nội dung CV.")
            with st.expander("Xem nội dung CV đã trích xuất"):
                st.text(cv_text[:10000])
                if len(cv_text) > 10000:
                    st.caption("Phần hiển thị đã được rút gọn.")

    st.subheader("Bước 2. Nhập mô tả công việc")
    job_text = st.text_area(
        "Nội dung mô tả công việc",
        height=220,
        placeholder=(
            "Ví dụ: Chúng tôi tuyển Marketing Intern. "
            "Ứng viên cần có kỹ năng nghiên cứu thị trường, "
            "Excel, giao tiếp, sáng tạo nội dung và làm việc nhóm..."
        )
    )

    st.caption("Bạn có thể sao chép mô tả tuyển dụng từ tin tuyển dụng thực tế.")

    analyze_button = st.button(
        "✨ Phân tích CV ngay",
        use_container_width=True
    )

    if analyze_button:
        if not uploaded_cv:
            st.warning("Vui lòng tải lên CV định dạng PDF trước.")
        elif not cv_text:
            st.warning(
                "Không trích xuất được văn bản từ CV. "
                "Nếu PDF là ảnh quét, ứng dụng hiện chưa hỗ trợ OCR."
            )
        elif not job_text.strip():
            st.warning("Vui lòng nhập mô tả công việc.")
        else:
            try:
                with st.spinner("AI đang phân tích CV, vui lòng chờ..."):
                    model = load_model()
                    result = analyze_cv(model, cv_text, job_text)

                st.session_state["cv_result"] = result
                st.session_state["cv_job_text"] = job_text
                st.session_state["cv_resume_text"] = cv_text

            except Exception as error:
                st.error(
                    "Có lỗi khi tải mô hình hoặc phân tích dữ liệu. "
                    "Vui lòng thử lại sau."
                )

    # Hiển thị kết quả đã phân tích
    if "cv_result" in st.session_state:
        result = st.session_state["cv_result"]

        st.divider()
        st.header("📊 Kết quả phân tích")

        st.caption(
            "Điểm tham khảo dựa trên độ tương đồng ngữ nghĩa giữa CV "
            "và mô tả công việc; không phải xác suất được tuyển dụng."
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric("Điểm phù hợp", f"{result['score']}/100")

        with metric2:
            st.metric("Kỹ năng phù hợp", len(result["matched"]))

        with metric3:
            st.metric("Kỹ năng còn thiếu", len(result["missing"]))

        # Biểu đồ điểm phù hợp
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=result["score"],
                number={"suffix": "/100"},
                title={"text": "Mức độ tương đồng ngữ nghĩa"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#8272D8"},
                    "bgcolor": "#F0EDFF",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "#FFE5E1"},
                        {"range": [40, 70], "color": "#FFF0D7"},
                        {"range": [70, 100], "color": "#DDF7E9"}
                    ]
                }
            )
        )

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=55, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#303653"}
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("✅ Kỹ năng phù hợp")

        if result["matched"]:
            matched_html = "".join(
                f'<span class="skill-good">{html.escape(skill)}</span>'
                for skill in result["matched"]
            )
            st.markdown(matched_html, unsafe_allow_html=True)
        else:
            st.write(
                "Chưa phát hiện kỹ năng trùng khớp trong danh sách từ khóa "
                "được hỗ trợ. Điều này không đồng nghĩa CV hoàn toàn không có kỹ năng phù hợp."
            )

        st.subheader("📌 Kỹ năng cần bổ sung hoặc làm rõ")

        if result["missing"]:
            missing_html = "".join(
                f'<span class="skill-missing">{html.escape(skill)}</span>'
                for skill in result["missing"]
            )
            st.markdown(missing_html, unsafe_allow_html=True)
            st.write(
                "Nếu bạn thực sự có các kỹ năng này, hãy bổ sung minh chứng "
                "cụ thể vào CV. Không nên ghi kỹ năng mà bạn chưa có."
            )
        else:
            st.success(
                "Không phát hiện kỹ năng nào còn thiếu trong danh sách từ khóa "
                "mà ứng dụng nhận diện."
            )

        st.subheader("💡 Gợi ý cải thiện CV")

        suggestions = []

        if result["missing"]:
            suggestions.append(
                "Xem lại các kỹ năng còn thiếu ở trên. Chỉ bổ sung những kỹ năng "
                "bạn thực sự có và đưa ra minh chứng bằng dự án, môn học hoặc kinh nghiệm."
            )

        if result["score"] < 50:
            suggestions.append(
                "Đọc lại mô tả công việc và làm rõ những kinh nghiệm, dự án "
                "và thành tích liên quan trực tiếp đến vị trí ứng tuyển."
            )
        elif result["score"] < 75:
            suggestions.append(
                "Tăng mức độ liên quan bằng cách mô tả rõ vai trò, công cụ đã dùng "
                "và kết quả đạt được trong các dự án hoặc công việc."
            )
        else:
            suggestions.append(
                "Tiếp tục kiểm tra tính chính xác của thông tin, thành tích "
                "và mức độ phù hợp của kinh nghiệm với yêu cầu tuyển dụng."
            )

        suggestions.append(
            "Ưu tiên các thành tích có số liệu cụ thể khi có căn cứ, "
            "chẳng hạn thời gian tiết kiệm, doanh số hoặc quy mô dự án."
        )

        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

        if result["skill_coverage"] is not None:
            st.subheader("📈 Mức độ bao phủ kỹ năng")

            st.progress(result["skill_coverage"] / 100)

            st.write(
                f"CV có {len(result['matched'])}/{len(result['job_skills'])} "
                f"kỹ năng được nhận diện trong mô tả công việc "
                f"({result['skill_coverage']}%)."
            )

        st.caption(
            "Lưu ý: Công cụ chỉ hỗ trợ tham khảo. Điểm ngữ nghĩa và từ khóa "
            "có thể chưa phản ánh đầy đủ năng lực thực tế của ứng viên."
        )


# =========================================================
# 10. TRANG HƯỚNG DẪN SỬ DỤNG
# =========================================================
elif page == "Hướng dẫn sử dụng":
    st.title("📘 Hướng dẫn sử dụng CVision AI")

    st.markdown("""
    ### Bước 1. Chuẩn bị CV
    - Sử dụng tệp PDF có thể chọn và sao chép văn bản.
    - Kiểm tra thông tin liên hệ, học vấn, kỹ năng và kinh nghiệm.

    ### Bước 2. Chuẩn bị mô tả công việc
    - Sao chép nội dung từ tin tuyển dụng.
    - Giữ lại các yêu cầu chính về kỹ năng và kinh nghiệm.

    ### Bước 3. Chạy phân tích
    - Mở mục **Phân tích CV**.
    - Tải CV và nhập mô tả công việc.
    - Nhấn **Phân tích CV ngay**.

    ### Bước 4. Đọc kết quả
    - **Điểm phù hợp:** mức tương đồng ngữ nghĩa tham khảo giữa hai nội dung.
    - **Kỹ năng phù hợp:** các kỹ năng ứng dụng nhận diện được ở cả hai bên.
    - **Kỹ năng còn thiếu:** các kỹ năng có trong mô tả công việc nhưng chưa được phát hiện trong CV.
    - **Gợi ý cải thiện:** các đề xuất giúp trình bày CV rõ ràng và sát vị trí ứng tuyển hơn.

    ### Lưu ý
    - Phiên bản hiện tại xử lý PDF có văn bản, chưa tích hợp OCR cho PDF chỉ chứa ảnh.
    - Việc nhận diện kỹ năng dựa trên danh sách từ khóa được lập trình sẵn.
    - Điểm phù hợp không phải xác suất trúng tuyển và không thay thế đánh giá của nhà tuyển dụng.
    """)


# =========================================================
# 11. TRANG GIỚI THIỆU
# =========================================================
elif page == "Giới thiệu":
    st.markdown("""
    <div class="hero">
        <h1>Về CVision AI</h1>
        <p>Công cụ hỗ trợ ứng viên hiểu rõ hơn mức độ phù hợp giữa CV và vị trí tuyển dụng.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Mục tiêu
    Hỗ trợ phân tích nội dung hồ sơ, đối chiếu yêu cầu tuyển dụng và xác định
    những điểm ứng viên nên xem xét cải thiện.

    ### 🧠 Công nghệ sử dụng
    - **Sentence Transformers:** biểu diễn văn bản thành vector để so sánh ngữ nghĩa.
    - **Cosine Similarity:** tính độ tương đồng giữa CV và mô tả công việc.
    - **PyMuPDF:** trích xuất văn bản từ tệp PDF.
    - **Scikit-learn:** hỗ trợ tính toán độ tương đồng.
    - **Plotly:** hiển thị biểu đồ kết quả.
    - **Streamlit:** xây dựng giao diện ứng dụng web.

    ### 🔒 Quyền riêng tư
    Chỉ tải lên CV mà bạn có quyền sử dụng. Tránh đưa thông tin cá nhân nhạy cảm
    vào các bản CV thử nghiệm công khai.
    """)

st.divider()
st.markdown(
    '<p class="small-note" style="text-align:center">'
    'CVision AI · Công cụ hỗ trợ phân tích hồ sơ ứng tuyển'
    '</p>',
    unsafe_allow_html=True
)


import re

import streamlit as st
import fitz
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# 1. PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="CVision AI | Your Career Companion",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 2. GEN Z DESIGN SYSTEM
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #282743;
    --muted: #77768F;
    --purple: #7967E8;
    --mint: #DDF8EC;
    --lavender: #F2EEFF;
    --sky: #E5F2FF;
    --peach: #FFF0E8;
    --white: #FFFFFF;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #FAF9FF;
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: rgba(250, 249, 255, 0.9);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #F0EDFF;
    border-right: 1px solid #E5E0FA;
}

[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

.brand {
    font-size: 25px;
    font-weight: 800;
    color: #5143B5;
    letter-spacing: -1px;
}

.brand-sub {
    color: #77708F;
    font-size: 12px;
    letter-spacing: 1.5px;
    font-weight: 700;
}

.side-note {
    background: white;
    border: 1px solid #E7E1FA;
    border-radius: 16px;
    padding: 16px;
    margin-top: 20px;
    color: #57536F;
    font-size: 13px;
    line-height: 1.8;
}

/* Hero */
.hero {
    background: linear-gradient(
        120deg, #EDE7FF 0%, #E1F8F0 60%, #E5F1FF 100%
    );
    border: 1px solid #E8E4F8;
    border-radius: 25px;
    padding: 32px;
    margin-bottom: 24px;
}

.eyebrow {
    color: #7160D1;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 800;
}

.hero h1 {
    color: #282743;
    font-size: clamp(30px, 4vw, 43px);
    font-weight: 800;
    letter-spacing: -1.7px;
    line-height: 1.15;
    margin: 12px 0;
}

.hero p {
    color: #66647F;
    font-size: 15px;
    line-height: 1.8;
    max-width: 700px;
}

.pill {
    display: inline-block;
    background: white;
    color: #5D50B7;
    border: 1px solid #DED7FF;
    padding: 7px 12px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
}

/* Section headings */
.section-heading {
    font-size: 21px;
    font-weight: 800;
    color: #302D4D;
    margin: 8px 0 5px;
    letter-spacing: -0.5px;
}

.section-caption {
    color: #828098;
    font-size: 13px;
    margin-bottom: 16px;
}

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid #EAE7F4;
    border-radius: 20px;
}

.metric-card {
    background: white;
    border: 1px solid #E9E5F5;
    border-radius: 18px;
    padding: 20px;
    min-height: 126px;
}

.metric-icon {
    font-size: 21px;
}

.metric-label {
    color: #817F99;
    font-size: 12px;
    margin-top: 8px;
}

.metric-value {
    color: #302D4D;
    font-size: 30px;
    font-weight: 800;
    margin-top: 5px;
}

.mini-card {
    background: #F7F5FF;
    border: 1px solid #EAE5FF;
    border-radius: 14px;
    padding: 15px;
    margin: 8px 0;
}

.mini-card p {
    color: #5E597B;
    font-size: 13px;
    line-height: 1.7;
    margin-bottom: 0;
}

.skill-tag {
    display: inline-block;
    background: #E4F8EF;
    color: #247956;
    border: 1px solid #C8EEDF;
    border-radius: 30px;
    padding: 7px 12px;
    margin: 4px 5px 4px 0;
    font-size: 12px;
    font-weight: 700;
}

.missing-tag {
    display: inline-block;
    background: #FFF0E8;
    color: #A45637;
    border: 1px solid #F9DACA;
    border-radius: 30px;
    padding: 7px 12px;
    margin: 4px 5px 4px 0;
    font-size: 12px;
    font-weight: 700;
}

/* Inputs */
.stTextInput input, .stTextArea textarea {
    background: #FFFFFF;
    border: 1px solid #E1DDF0;
    border-radius: 12px;
    color: #302D4D;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #9585F0;
    box-shadow: 0 0 0 2px #EDE8FF;
}

[data-testid="stFileUploader"] {
    background: #F8F6FF;
    border: 1.5px dashed #B7A9F4;
    border-radius: 15px;
    padding: 12px;
}

/* Buttons */
.stButton > button {
    background: #7967E8;
    color: white;
    border: 1px solid #7967E8;
    border-radius: 12px;
    font-weight: 700;
    min-height: 46px;
    transition: all .2s ease;
}

.stButton > button:hover {
    background: #6552D4;
    color: white;
    border-color: #6552D4;
    transform: translateY(-1px);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #F0EDFA;
    border-radius: 13px;
    padding: 6px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: #67637F;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: white !important;
    color: #6251CF !important;
}

/* Footer */
.footer {
    border-top: 1px solid #EAE6F5;
    margin-top: 35px;
    padding-top: 18px;
    color: #89869F;
    font-size: 12px;
    text-align: center;
}

h1, h2, h3 {
    color: #302D4D;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

@media (max-width: 768px) {
    .hero {
        padding: 22px;
    }
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. SKILL DICTIONARY
# ==================================================
SKILLS = {
    "Python": ["python"],
    "SQL": ["sql", "mysql", "postgresql", "sqlite"],
    "Excel": ["excel", "spreadsheet"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Machine Learning": ["machine learning", "hoc may", "học máy"],
    "Deep Learning": ["deep learning", "hoc sau", "học sâu"],
    "Data Analysis": [
        "data analysis", "data analytics",
        "phan tich du lieu", "phân tích dữ liệu"
    ],
    "Statistics": ["statistics", "thong ke", "thống kê"],
    "Communication": [
        "communication", "giao tiep", "giao tiếp"
    ],
    "Teamwork": ["teamwork", "team work", "lam viec nhom", "làm việc nhóm"],
    "Problem Solving": [
        "problem solving", "giai quyet van de",
        "giải quyết vấn đề"
    ],
    "Project Management": [
        "project management", "quan ly du an", "quản lý dự án"
    ],
    "Marketing": ["marketing", "tiep thi", "tiếp thị"],
    "SEO": ["seo", "search engine optimization"],
    "Content Writing": [
        "content writing", "copywriting", "viet noi dung", "viết nội dung"
    ],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Data Visualization": [
        "data visualization", "truc quan hoa du lieu",
        "trực quan hóa dữ liệu"
    ],
    "Artificial Intelligence": [
        "artificial intelligence", "tri tue nhan tao",
        "trí tuệ nhân tạo"
    ],
    "Leadership": ["leadership", "lanh dao", "lãnh đạo"],
    "English": ["english", "tieng anh", "tiếng anh"],
    "Customer Service": [
        "customer service", "cham soc khach hang",
        "chăm sóc khách hàng"
    ],
    "Sales": ["sales", "ban hang", "bán hàng"],
}


# ==================================================
# 4. TEXT PROCESSING
# ==================================================
def extract_pdf_text(file_bytes):
    parts = []

    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page in pdf:
            parts.append(page.get_text())

    return "\n".join(parts).strip()


def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_skills(text):
    text = normalize_text(text)
    found = []

    for skill, aliases in SKILLS.items():
        for alias in aliases:
            pattern = (
                r"(?<!\w)" + re.escape(alias.lower()) + r"(?!\w)"
            )
            if re.search(pattern, text):
                found.append(skill)
                break

    return sorted(set(found))


# ==================================================
# 5. AI MODEL
# ==================================================
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

    similarity = float(
        cosine_similarity(
            [embeddings[0]], [embeddings[1]]
        )[0][0]
    )

    # Similarity score, not probability of being hired
    score = round(max(0, min(100, similarity * 100)))

    cv_skills = extract_skills(cv_text)
    job_skills = extract_skills(job_text)

    matched = sorted(set(cv_skills) & set(job_skills))
    missing = sorted(set(job_skills) - set(cv_skills))

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


# ==================================================
# 6. SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown(
        '<div class="brand">✦ CVision AI</div>'
        '<div class="brand-sub">YOUR CAREER COMPANION</div>',
        unsafe_allow_html=True
    )

    st.write("")
    page = st.radio(
        "MENU",
        ["✨ Phân tích CV", "📘 Hướng dẫn sử dụng"],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown("""
    <div class="side-note">
        <b>✦ Career, but smarter.</b><br><br>
        Phân tích hồ sơ ứng tuyển bằng AI,
        tìm kỹ năng phù hợp và xác định
        những điểm bạn có thể cải thiện.<br><br>
        <b>AI-powered · Student-friendly</b>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.caption("CVision AI · MVP Edition")


# ==================================================
# 7. HERO HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <span class="pill">✦ YOUR CAREER, YOUR STORY</span>
    <h1>Your next career move<br>starts here.</h1>
    <p>
        Biến CV của bạn thành lợi thế cạnh tranh.
        Tải hồ sơ, chọn công việc mục tiêu và khám phá
        mức độ tương đồng cùng những kỹ năng cần phát triển.
    </p>
    <span class="eyebrow">AI CAREER INSIGHTS · MADE FOR YOU</span>
</div>
""", unsafe_allow_html=True)


# ==================================================
# 8. GUIDE PAGE
# ==================================================
if page == "📘 Hướng dẫn sử dụng":
    st.markdown(
        '<div class="section-heading">Cách sử dụng CVision AI</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-caption">'
        'Ba bước để bắt đầu hành trình ứng tuyển thông minh hơn.'
        '</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">📄</div>
            <h3>01. Upload</h3>
            <p>Tải lên CV PDF có thể trích xuất văn bản.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">💼</div>
            <h3>02. Match</h3>
            <p>Nhập mô tả công việc bạn muốn ứng tuyển.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-icon">🚀</div>
            <h3>03. Improve</h3>
            <p>Xem báo cáo, đối chiếu kỹ năng và tìm điểm cần cải thiện.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Lưu ý: CVision AI hỗ trợ tham khảo, không quyết định "
        "khả năng trúng tuyển. Hệ thống hiện chưa đọc được "
        "nội dung CV chỉ chứa hình ảnh scan."
    )


# ==================================================
# 9. ANALYSIS PAGE
# ==================================================
else:
    st.markdown(
        '<div class="section-heading">Let’s find your match ✨</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-caption">'
        'Bắt đầu bằng CV của bạn và mô tả công việc mục tiêu.'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.markdown("### 📄 Your resume")
            st.markdown(
                '<div class="section-caption">'
                'Tải hồ sơ ứng tuyển của bạn'
                '</div>',
                unsafe_allow_html=True
            )

            uploaded_cv = st.file_uploader(
                "Chọn file CV (PDF)",
                type=["pdf"],
                help="Dùng PDF có văn bản, không phải ảnh scan."
            )

            if uploaded_cv:
                st.success(
                    f"✓ Đã tải: {uploaded_cv.name}"
                )

            st.markdown("""
            <div class="mini-card">
                <p>
                <b>Quick tip 💡</b><br>
                CV có cấu trúc rõ ràng và nội dung văn bản
                sẽ giúp hệ thống phân tích tốt hơn.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            st.markdown("### 💼 Your dream role")
            st.markdown(
                '<div class="section-caption">'
                'Bạn đang hướng đến công việc nào?'
                '</div>',
                unsafe_allow_html=True
            )

            job_title = st.text_input(
                "Tên vị trí",
                placeholder="VD: Marketing Intern"
            )

            job_description = st.text_area(
                "Job description",
                height=165,
                placeholder=(
                    "Dán mô tả công việc vào đây...\n\n"
                    "Ví dụ: Yêu cầu Excel, SQL, phân tích dữ liệu..."
                )
            )

            st.caption(
                "Mẹo: Bao gồm trách nhiệm, kỹ năng và yêu cầu tuyển dụng."
            )

    st.write("")

    analyze_button = st.button(
        "✨ Analyze my CV",
        use_container_width=True
    )

    if analyze_button:
        if not uploaded_cv:
            st.error("Bạn hãy tải CV PDF trước nhé.")
            st.stop()

        if len(job_description.strip()) < 30:
            st.error(
                "Vui lòng nhập mô tả công việc dài ít nhất 30 ký tự."
            )
            st.stop()

        try:
            with st.spinner(
                "AI đang đọc hồ sơ và phân tích độ tương đồng..."
            ):
                cv_text = extract_pdf_text(uploaded_cv.getvalue())

                if len(cv_text) < 50:
                    st.error(
                        "Không đọc được đủ văn bản từ PDF. "
                        "Hãy thử một file PDF có thể chọn và sao chép chữ."
                    )
                    st.stop()

                result = analyze_match(
                    cv_text, job_description
                )

            st.session_state["analysis_result"] = result
            st.session_state["job_title"] = (
                job_title.strip() or "Vị trí chưa đặt tên"
            )
            st.session_state["cv_filename"] = uploaded_cv.name

        except Exception as error:
            st.error(
                "Phân tích chưa thành công. "
                "Hãy kiểm tra kết nối và nhật ký ứng dụng."
            )
            st.code(str(error))

    # Render saved result after Streamlit reruns
    if "analysis_result" in st.session_state:
        result = st.session_state["analysis_result"]

        st.write("")
        st.markdown("---")

        st.markdown(
            '<div class="section-heading">'
            'Your career insights ✨'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-caption">'
            'Báo cáo phân tích dựa trên CV và mô tả công việc đã nhập.'
            '</div>',
            unsafe_allow_html=True
        )

        st.caption(
            f"CV: {st.session_state['cv_filename']} · "
            f"Role: {st.session_state['job_title']}"
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-label">SEMANTIC SIMILARITY</div>
                <div class="metric-value">{result['score']}<span style="font-size:15px">/100</span></div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🌱</div>
                <div class="metric-label">MATCHED SKILLS</div>
                <div class="metric-value">{len(result['matched'])}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">💡</div>
                <div class="metric-label">SKILLS TO REVIEW</div>
                <div class="metric-value">{len(result['missing'])}</div>
            </div>
            """, unsafe_allow_html=True)

        st.info(
            "Điểm tương đồng là mức độ gần nhau giữa biểu diễn "
            "ngữ nghĩa của CV và mô tả công việc, không phải "
            "xác suất trúng tuyển."
        )

        tab1, tab2, tab3 = st.tabs([
            "🌿 Skill match",
            "📊 Analytics",
            "🚀 Improvement tips"
        ])

        with tab1:
            good, gap = st.columns(2)

            with good:
                st.markdown("### 🌿 Skills found in both")

                if result["matched"]:
                    tags = "".join(
                        f'<span class="skill-tag">✓ {skill}</span>'
                        for skill in result["matched"]
                    )
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.write(
                        "Chưa tìm thấy kỹ năng trùng khớp "
                        "theo bộ từ khóa hiện tại."
                    )

            with gap:
                st.markdown("### 🧡 Skills to review")

                if result["missing"]:
                    tags = "".join(
                        f'<span class="missing-tag">＋ {skill}</span>'
                        for skill in result["missing"]
                    )
                    st.markdown(tags, unsafe_allow_html=True)
                elif result["job_skills"]:
                    st.success(
                        "Đã nhận diện các kỹ năng yêu cầu "
                        "trong bộ từ khóa của hệ thống."
                    )
                else:
                    st.write(
                        "Chưa phát hiện kỹ năng từ mô tả công việc. "
                        "Thử ghi rõ tên kỹ năng."
                    )

            st.markdown("### 📝 Skills detected in your CV")
            if result["cv_skills"]:
                st.write(", ".join(result["cv_skills"]))
            else:
                st.write(
                    "Chưa nhận diện được kỹ năng theo bộ từ khóa."
                )

            st.markdown("### 🔎 Skills detected in the job description")
            if result["job_skills"]:
                st.write(", ".join(result["job_skills"]))
            else:
                st.write("Chưa nhận diện được kỹ năng.")

        with tab2:
            labels = ["Độ tương đồng ngữ nghĩa"]
            values = [result["score"]]
            colors = ["#7967E8"]

            if result["coverage"] is not None:
                labels.append("Độ bao phủ kỹ năng")
                values.append(result["coverage"])
                colors.append("#52BFA1")

            fig = go.Figure(
                go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    text=[f"{value}%" for value in values],
                    textposition="auto",
                    marker_color=colors,
                    marker_line_width=0,
                )
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="DM Sans",
                    color="#45415F"
                ),
                xaxis=dict(
                    title="Điểm (%)",
                    range=[0, 100],
                    showgrid=True,
                    gridcolor="#EEEAF7"
                ),
                yaxis=dict(
                    title="",
                    autorange="reversed"
                ),
                height=250,
                margin=dict(l=10, r=20, t=20, b=35),
                showlegend=False,
            )

            st.plotly_chart(
                fig, use_container_width=True
            )

            st.markdown("""
            <div class="mini-card">
                <p>
                <b>How to read this chart</b><br>
                Semantic similarity đo độ tương đồng ngữ nghĩa giữa
                CV và mô tả công việc. Skill coverage đo tỷ lệ kỹ năng
                yêu cầu được tìm thấy trong CV theo bộ từ khóa.
                Hai chỉ số có cách tính khác nhau.
                </p>
            </div>
            """, unsafe_allow_html=True)

            if result["coverage"] is not None:
                st.metric(
                    "Skill coverage",
                    f"{result['coverage']}%"
                )
            else:
                st.caption(
                    "Chưa tính được skill coverage vì chưa phát hiện "
                    "kỹ năng trong mô tả công việc."
                )

        with tab3:
            suggestions = []

            if result["missing"]:
                suggestions.append(
                    "Xem lại các kỹ năng còn thiếu. Chỉ bổ sung vào CV "
                    "nếu bạn thực sự có kiến thức hoặc kinh nghiệm đó."
                )
                suggestions.append(
                    "Nếu đã có kinh nghiệm liên quan, hãy trình bày "
                    "qua dự án, nhiệm vụ và kết quả cụ thể."
                )

            if result["score"] < 50:
                suggestions.append(
                    "Làm rõ các kinh nghiệm liên quan đến vị trí. "
                    "Điểm tương đồng thấp không đồng nghĩa bạn "
                    "không đủ năng lực."
                )

            suggestions.append(
                "Ưu tiên nêu bật các dự án, kỹ năng và thành tích "
                "liên quan trực tiếp đến công việc."
            )
            suggestions.append(
                "Kiểm tra thông tin liên hệ, học vấn, lỗi chính tả "
                "và bố cục trước khi gửi CV."
            )

            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(
                    f"""
                    <div class="mini-card">
                        <b style="color:#6552D4">STEP {i:02d}</b>
                        <p>{suggestion}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.caption(
            "CVision AI là công cụ hỗ trợ tham khảo. Bộ từ khóa "
            "có thể bỏ sót kỹ năng được diễn đạt theo cách khác. "
            "Không dùng điểm số để tự động loại ứng viên."
        )


# ==================================================
# 10. FOOTER
# ==================================================
st.markdown("""
<div class="footer">
    ✦ CVision AI · Your Career Companion<br>
    AI-powered career insights for a brighter future.
</div>
""", unsafe_allow_html=True)

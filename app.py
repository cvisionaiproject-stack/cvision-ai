
import re
import io
from datetime import datetime

import streamlit as st
import pandas as pd

# Các thư viện đọc PDF và ảnh được nạp an toàn
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None


# =========================================================
# 1. CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="CVision AI | Career Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

NAVY = "#10243A"
NAVY_LIGHT = "#183A56"
TEAL = "#16B8A6"
BG = "#F4F7FA"
TEXT = "#243447"
MUTED = "#66788A"

# =========================================================
# 2. GIAO DIỆN
# =========================================================
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG};
        color: {TEXT};
    }}
    [data-testid="stSidebar"] {{
        background: {NAVY};
    }}
    [data-testid="stSidebar"] * {{
        color: #F2F7FA;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        padding: 7px 10px;
        border-radius: 8px;
    }}
    .brand {{
        font-size: 27px;
        font-weight: 800;
        color: {NAVY};
        margin-bottom: 2px;
    }}
    .subtitle {{
        color: {MUTED};
        margin-bottom: 22px;
    }}
    .hero {{
        padding: 24px;
        border-radius: 16px;
        background: linear-gradient(120deg, {NAVY}, {NAVY_LIGHT});
        color: white;
        margin-bottom: 22px;
    }}
    .hero h2 {{
        color: white;
        margin: 0 0 8px 0;
    }}
    .hero p {{
        color: #D8E7F1;
        margin: 0;
    }}
    .metric-card {{
        background: white;
        border: 1px solid #E2EAF0;
        padding: 18px;
        border-radius: 13px;
        min-height: 100px;
    }}
    .section-title {{
        color: {NAVY};
        font-size: 21px;
        font-weight: 750;
        margin: 8px 0 12px 0;
    }}
    div.stButton > button {{
        background: {TEAL};
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 650;
        min-height: 42px;
    }}
    div.stButton > button:hover {{
        background: #109A8C;
        color: white;
    }}
    div[data-testid="stFileUploader"] {{
        background: white;
        border-radius: 12px;
        padding: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. TRẠNG THÁI ỨNG DỤNG
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""

if "cv_filename" not in st.session_state:
    st.session_state.cv_filename = ""

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "rewrite_output" not in st.session_state:
    st.session_state.rewrite_output = ""

if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = []


# =========================================================
# 4. HÀM XỬ LÝ CV
# =========================================================
SKILLS = [
    "python", "excel", "sql", "power bi", "tableau",
    "machine learning", "data analysis", "data analytics",
    "marketing", "digital marketing", "seo", "content",
    "sales", "communication", "teamwork", "leadership",
    "project management", "research", "statistics",
    "financial analysis", "accounting", "customer service",
    "problem solving", "critical thinking", "english",
    "java", "javascript", "html", "css", "figma",
    "canva", "presentation", "negotiation"
]

CV_SECTIONS = {
    "Thông tin liên hệ": [
        "email", "phone", "điện thoại", "contact", "linkedin"
    ],
    "Học vấn": [
        "education", "học vấn", "university", "đại học", "degree"
    ],
    "Kinh nghiệm": [
        "experience", "kinh nghiệm", "employment", "work history"
    ],
    "Kỹ năng": [
        "skills", "kỹ năng", "technical skills", "soft skills"
    ],
    "Dự án": [
        "project", "dự án", "portfolio"
    ],
    "Chứng chỉ": [
        "certification", "chứng chỉ", "certificate"
    ]
}


def extract_text(uploaded_file):
    """Đọc nội dung từ PDF hoặc ảnh."""
    filename = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if filename.endswith(".pdf"):
        if fitz is None:
            raise RuntimeError(
                "Chưa có PyMuPDF. Hãy kiểm tra requirements.txt."
            )

        document = fitz.open(stream=raw, filetype="pdf")
        pages = [page.get_text() for page in document]
        text = "\n".join(pages).strip()

        # Nếu PDF là bản scan, thử OCR khi thư viện sẵn có
        if len(text) < 30 and Image is not None and pytesseract is not None:
            ocr_pages = []
            for page in document:
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                try:
                    ocr_pages.append(pytesseract.image_to_string(
                        image, lang="vie+eng"
                    ))
                except Exception:
                    ocr_pages.append(
                        pytesseract.image_to_string(image, lang="eng")
                    )
            text = "\n".join(ocr_pages).strip()

        return text

    if filename.endswith((".png", ".jpg", ".jpeg")):
        if Image is None:
            raise RuntimeError("Chưa có Pillow để đọc ảnh.")

        if pytesseract is None:
            raise RuntimeError(
                "Chưa có pytesseract. Hãy tải CV dưới dạng PDF có văn bản "
                "hoặc bổ sung cấu hình OCR."
            )

        image = Image.open(io.BytesIO(raw))
        try:
            return pytesseract.image_to_string(image, lang="vie+eng").strip()
        except Exception:
            return pytesseract.image_to_string(image, lang="eng").strip()

    raise ValueError("Định dạng chưa được hỗ trợ.")


def find_skills(text):
    text_lower = text.lower()
    return [skill for skill in SKILLS if skill in text_lower]


def find_sections(text):
    text_lower = text.lower()
    found = {}

    for section, keywords in CV_SECTIONS.items():
        found[section] = any(
            keyword in text_lower for keyword in keywords
        )

    return found


def analyze_cv(text):
    skills = find_skills(text)
    sections = find_sections(text)
    word_count = len(re.findall(r"\b[\w+#.-]+\b", text))

    email_found = bool(
        re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    )
    phone_found = bool(
        re.search(r"(\+?\d[\d\s().-]{7,}\d)", text)
    )

    section_score = (
        sum(sections.values()) / len(sections) * 60
    )
    contact_score = (
        int(email_found) + int(phone_found)
    ) / 2 * 15
    length_score = min(word_count / 350, 1) * 15
    skill_score = min(len(skills) / 8, 1) * 10

    score = round(
        min(section_score + contact_score + length_score + skill_score, 100)
    )

    strengths = []
    improvements = []

    if sections["Học vấn"]:
        strengths.append("Có thông tin học vấn.")
    else:
        improvements.append("Bổ sung mục học vấn.")

    if sections["Kinh nghiệm"]:
        strengths.append("Có mục kinh nghiệm làm việc.")
    else:
        improvements.append(
            "Thêm kinh nghiệm thực tập, làm việc hoặc hoạt động liên quan."
        )

    if sections["Kỹ năng"]:
        strengths.append("Có đề cập đến kỹ năng.")
    else:
        improvements.append("Tạo mục kỹ năng và nêu kỹ năng phù hợp vị trí.")

    if sections["Dự án"]:
        strengths.append("Có đề cập đến dự án hoặc portfolio.")
    else:
        improvements.append(
            "Bổ sung dự án học tập hoặc dự án cá nhân kèm kết quả cụ thể."
        )

    if email_found:
        strengths.append("Phát hiện địa chỉ email.")
    else:
        improvements.append("Bổ sung email liên hệ chuyên nghiệp.")

    if phone_found:
        strengths.append("Phát hiện số điện thoại.")
    else:
        improvements.append("Kiểm tra và bổ sung số điện thoại.")

    if word_count < 150:
        improvements.append(
            "Nội dung CV khá ngắn; cân nhắc bổ sung thành tích và minh chứng."
        )

    if not skills:
        improvements.append(
            "Nêu rõ kỹ năng chuyên môn bằng từ khóa phù hợp công việc."
        )

    return {
        "score": score,
        "word_count": word_count,
        "skills": skills,
        "sections": sections,
        "strengths": strengths,
        "improvements": improvements,
        "email_found": email_found,
        "phone_found": phone_found
    }


def similarity_score(cv_text, job_text):
    """Tính độ tương đồng văn bản bằng TF-IDF."""
    if TfidfVectorizer is None or cosine_similarity is None:
        return None

    if not cv_text.strip() or not job_text.strip():
        return None

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000
        )
        matrix = vectorizer.fit_transform([cv_text, job_text])
        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return round(float(score) * 100, 1)
    except Exception:
        return None


def save_history(filename, result):
    st.session_state.history.insert(0, {
        "Thời gian": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Tên tệp": filename,
        "Điểm checklist": result["score"],
        "Số kỹ năng": len(result["skills"])
    })
    st.session_state.history = st.session_state.history[:50]


def page_header(title, subtitle):
    st.markdown(f'<div class="brand">CVision AI</div>',
                unsafe_allow_html=True)
    st.markdown(f"### {title}")
    st.markdown(
        f'<div class="subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


# =========================================================
# 5. THANH MENU
# =========================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:12px 4px 22px 4px;">
            <div style="font-size:30px;font-weight:800;">
                CVision <span style="color:{TEAL};">AI</span>
            </div>
            <div style="font-size:12px;color:#C2D2DF;">
                CAREER INTELLIGENCE PLATFORM
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### ĐIỀU HƯỚNG")

    page = st.radio(
        "Chọn chức năng",
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
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("AI hỗ trợ định hướng nghề nghiệp")
    st.caption("Kết quả chỉ mang tính tham khảo.")


# =========================================================
# 6. TỔNG QUAN
# =========================================================
if page == "Tổng quan":
    page_header(
        "Career Intelligence Dashboard",
        "Phân tích CV, đối chiếu yêu cầu tuyển dụng và chuẩn bị phỏng vấn."
    )

    st.markdown(
        f"""
        <div class="hero">
            <h2>Biến CV thành lợi thế nghề nghiệp</h2>
            <p>
                Đánh giá cấu trúc CV, nhận diện kỹ năng và tìm những điểm
                cần cải thiện trước khi ứng tuyển.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            '<div class="metric-card"><b>01 · CV Analysis</b>'
            '<p>Kiểm tra nội dung và cấu trúc CV.</p></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            '<div class="metric-card"><b>02 · Job Matching</b>'
            '<p>So sánh nội dung CV với mô tả công việc.</p></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            '<div class="metric-card"><b>03 · Interview Prep</b>'
            '<p>Chuẩn bị câu hỏi phỏng vấn theo vị trí.</p></div>',
            unsafe_allow_html=True
        )

    st.markdown("### Bắt đầu sử dụng")

    if st.button("📄 Phân tích CV ngay", use_container_width=True):
        st.session_state["go_to_analysis"] = True
        st.rerun()

    if st.session_state.get("go_to_analysis", False):
        st.info("Chọn mục **Phân tích CV** trong thanh menu bên trái.")

    st.markdown("### Quy trình hoạt động")
    st.write(
        "1. Tải CV lên hệ thống.\n"
        "2. Trích xuất văn bản từ PDF hoặc ảnh.\n"
        "3. Phân tích các mục, kỹ năng và thông tin liên hệ.\n"
        "4. Đối chiếu với mô tả công việc nếu cần.\n"
        "5. Tham khảo gợi ý cải thiện và câu hỏi phỏng vấn."
    )


# =========================================================
# 7. PHÂN TÍCH CV
# =========================================================
elif page == "Phân tích CV":
    page_header(
        "Phân tích CV",
        "Tải lên CV để kiểm tra nội dung, kỹ năng và các mục quan trọng."
    )

    uploaded_file = st.file_uploader(
        "Tải CV của bạn lên",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Hỗ trợ PDF, PNG và JPG. PDF có văn bản thường cho kết quả tốt hơn.",
        key="cv_upload"
    )

    if uploaded_file is not None:
        st.success(f"Đã chọn tệp: {uploaded_file.name}")

        if st.button("🔍 Phân tích CV", use_container_width=True):
            try:
                with st.spinner("Đang đọc và phân tích CV..."):
                    extracted = extract_text(uploaded_file)

                    if not extracted.strip():
                        st.error(
                            "Không trích xuất được văn bản. "
                            "Hãy thử PDF có văn bản hoặc ảnh rõ nét."
                        )
                    else:
                        result = analyze_cv(extracted)
                        st.session_state.cv_text = extracted
                        st.session_state.cv_filename = uploaded_file.name
                        st.session_state.analysis = result
                        save_history(uploaded_file.name, result)
                        st.success("Phân tích hoàn tất.")

            except Exception as exc:
                st.error(f"Không thể xử lý tệp: {exc}")

    if st.session_state.analysis:
        result = st.session_state.analysis

        st.markdown("---")
        st.markdown("### Kết quả phân tích")

        c1, c2, c3 = st.columns(3)
        c1.metric("Điểm checklist", f'{result["score"]}/100')
        c2.metric("Số từ ước tính", result["word_count"])
        c3.metric("Kỹ năng nhận diện", len(result["skills"]))

        st.caption(
            "Điểm checklist là chỉ số tham khảo dựa trên cấu trúc và "
            "mức độ đầy đủ của CV; không phải xác suất được tuyển dụng."
        )

        st.markdown("### Kiểm tra các mục trong CV")

        section_df = pd.DataFrame([
            {
                "Mục": section,
                "Trạng thái": "Đã phát hiện" if found else "Chưa phát hiện"
            }
            for section, found in result["sections"].items()
        ])
        st.dataframe(section_df, use_container_width=True, hide_index=True)

        left, right = st.columns(2)

        with left:
            st.markdown("#### Điểm tích cực")
            if result["strengths"]:
                for item in result["strengths"]:
                    st.success(item)
            else:
                st.info("Chưa có đủ thông tin để xác định.")

        with right:
            st.markdown("#### Gợi ý cải thiện")
            if result["improvements"]:
                for item in result["improvements"]:
                    st.warning(item)
            else:
                st.info("Hãy kiểm tra mức độ phù hợp với từng vị trí.")

        st.markdown("### Kỹ năng được nhận diện")
        if result["skills"]:
            st.write(" · ".join(result["skills"]))
        else:
            st.info(
                "Chưa phát hiện từ khóa kỹ năng trong danh sách hiện có. "
                "Điều này không đồng nghĩa CV không có kỹ năng."
            )

        with st.expander("Xem văn bản trích xuất từ CV"):
            st.text_area(
                "Nội dung đọc được",
                value=st.session_state.cv_text,
                height=250
            )

        st.download_button(
            "⬇️ Tải văn bản CV đã trích xuất",
            data=st.session_state.cv_text,
            file_name="cv_text.txt",
            mime="text/plain"
        )


# =========================================================
# 8. SO SÁNH VỊ TRÍ
# =========================================================
elif page == "So sánh vị trí":
    page_header(
        "So sánh CV với vị trí tuyển dụng",
        "Đối chiếu nội dung CV với mô tả công việc (Job Description)."
    )

    st.info(
        "Trước tiên hãy phân tích CV, hoặc dán nội dung CV trực tiếp "
        "vào ô bên dưới."
    )

    cv_input = st.text_area(
        "Nội dung CV",
        value=st.session_state.cv_text,
        height=180,
        placeholder="Dán nội dung CV vào đây..."
    )

    job_input = st.text_area(
        "Mô tả công việc (Job Description)",
        height=180,
        placeholder="Dán yêu cầu tuyển dụng vào đây..."
    )

    if st.button("📊 So sánh mức độ phù hợp", use_container_width=True):
        if not cv_input.strip() or not job_input.strip():
            st.warning("Vui lòng cung cấp cả nội dung CV và mô tả công việc.")
        else:
            score = similarity_score(cv_input, job_input)

            if score is None:
                st.error(
                    "Không thể tính độ tương đồng. "
                    "Hãy kiểm tra thư viện scikit-learn trong requirements.txt."
                )
            else:
                st.metric("Độ tương đồng văn bản", f"{score}%")
                st.progress(int(score))

                cv_skills = set(find_skills(cv_input))
                job_skills = set(find_skills(job_input))
                matched = sorted(cv_skills & job_skills)
                missing = sorted(job_skills - cv_skills)

                st.markdown("#### Kỹ năng trùng khớp")
                st.write(", ".join(matched) if matched else "Chưa phát hiện.")

                st.markdown("#### Kỹ năng trong JD chưa thấy ở CV")
                st.write(", ".join(missing) if missing else "Chưa phát hiện.")

                st.caption(
                    "Độ tương đồng TF-IDF phản ánh mức độ giống nhau của "
                    "văn bản, không phải xác suất trúng tuyển hay đánh giá "
                    "đầy đủ năng lực ứng viên."
                )


# =========================================================
# 9. GỢI Ý VIẾT LẠI CV
# =========================================================
elif page == "Gợi ý viết lại CV":
    page_header(
        "Gợi ý cải thiện CV",
        "Tạo bản nháp gợi ý dựa trên nội dung CV hiện có."
    )

    source_text = st.text_area(
        "Nội dung CV cần cải thiện",
        value=st.session_state.cv_text,
        height=220,
        placeholder="Dán nội dung CV hoặc phân tích CV trước..."
    )

    target_role = st.text_input(
        "Vị trí muốn ứng tuyển",
        placeholder="Ví dụ: Marketing Intern, Data Analyst Intern"
    )

    if st.button("✨ Tạo gợi ý", use_container_width=True):
        if not source_text.strip():
            st.warning("Vui lòng tải CV hoặc dán nội dung CV trước.")
        else:
            skills = find_skills(source_text)
            sections = find_sections(source_text)

            suggestions = [
                f"# Gợi ý cải thiện CV cho vị trí: {target_role or 'chưa xác định'}",
                "",
                "## 1. Bố cục và nội dung",
            ]

            for section, found in sections.items():
                status = "Đã phát hiện" if found else "Chưa phát hiện"
                suggestions.append(f"- {section}: {status}.")

            suggestions.extend([
                "",
                "## 2. Kỹ năng nhận diện",
                ", ".join(skills) if skills else
                "Chưa nhận diện được kỹ năng từ danh sách từ khóa hiện có.",
                "",
                "## 3. Gợi ý hành động",
                "- Mô tả thành tích bằng số liệu khi có bằng chứng thực tế.",
                "- Ưu tiên kinh nghiệm và dự án liên quan vị trí ứng tuyển.",
                "- Dùng động từ hành động rõ ràng, tránh mô tả chung chung.",
                "- Chỉ thêm kỹ năng và kinh nghiệm mà bạn thực sự có.",
                "- Kiểm tra lỗi chính tả, ngày tháng và thông tin liên hệ.",
                "",
                "## 4. Mẫu viết thành tích",
                "Trước: Tham gia làm dự án marketing.",
                "Sau: Tham gia [tên dự án], phụ trách [nhiệm vụ cụ thể], "
                "đạt [kết quả đo lường được].",
                "",
                "Hãy thay phần trong ngoặc bằng thông tin thực tế; "
                "không sử dụng số liệu chưa được xác minh."
            ])

            st.session_state.rewrite_output = "\n".join(suggestions)

    if st.session_state.rewrite_output:
        st.markdown("### Bản gợi ý")
        st.markdown(st.session_state.rewrite_output)

        st.download_button(
            "⬇️ Tải gợi ý dưới dạng Markdown",
            data=st.session_state.rewrite_output,
            file_name="goi_y_cai_thien_cv.md",
            mime="text/markdown"
        )


# =========================================================
# 10. LUYỆN PHỎNG VẤN
# =========================================================
elif page == "Luyện phỏng vấn":
    page_header(
        "Luyện phỏng vấn",
        "Tham khảo câu hỏi để chuẩn bị cho vị trí ứng tuyển."
    )

    role = st.text_input(
        "Vị trí ứng tuyển",
        placeholder="Ví dụ: Business Analyst Intern"
    )

    jd = st.text_area(
        "Mô tả công việc (không bắt buộc)",
        height=140,
        placeholder="Dán mô tả công việc để cá nhân hóa câu hỏi..."
    )

    if st.button("🎯 Tạo bộ câu hỏi", use_container_width=True):
        questions = [
            f"Bạn hãy giới thiệu bản thân và lý do ứng tuyển vị trí {role or 'này'}.",
            "Dự án hoặc kinh nghiệm nào thể hiện rõ nhất năng lực của bạn?",
            "Bạn xử lý thế nào khi gặp một nhiệm vụ mới chưa có kinh nghiệm?",
            "Hãy kể về một lần làm việc nhóm gặp khó khăn và cách bạn giải quyết.",
            "Điểm mạnh nào của bạn phù hợp với vị trí này?",
            "Bạn đang muốn cải thiện kỹ năng nào và thực hiện điều đó ra sao?",
            "Bạn có câu hỏi gì dành cho nhà tuyển dụng?"
        ]

        jd_skills = find_skills(jd)
        if jd_skills:
            questions.insert(
                2,
                "Bạn đã sử dụng hoặc sẽ học các kỹ năng sau như thế nào: "
                + ", ".join(jd_skills[:6]) + "?"
            )

        st.session_state.interview_questions = questions

    for i, question in enumerate(
        st.session_state.interview_questions, start=1
    ):
        with st.expander(f"Câu {i}: {question}"):
            st.text_area(
                "Câu trả lời của bạn",
                key=f"answer_{i}",
                placeholder="Ghi ý chính cho câu trả lời...",
                height=100
            )


# =========================================================
# 11. LỊCH SỬ
# =========================================================
elif page == "Lịch sử phân tích":
    page_header(
        "Lịch sử phân tích",
        "Xem các lượt phân tích CV trong phiên làm việc hiện tại."
    )

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Xuất lịch sử CSV",
            data=history_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="lich_su_phan_tich.csv",
            mime="text/csv"
        )

        if st.button("Xóa lịch sử trong phiên này"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Chưa có lượt phân tích CV nào trong phiên làm việc này.")

    st.caption(
        "Lịch sử được lưu trong session hiện tại; có thể mất khi phiên "
        "ứng dụng kết thúc hoặc được khởi động lại."
    )


# =========================================================
# 12. HƯỚNG DẪN
# =========================================================
elif page == "Hướng dẫn sử dụng":
    page_header(
        "Hướng dẫn sử dụng",
        "Các bước sử dụng CVision AI."
    )

    st.markdown("""
    **Bước 1 — Phân tích CV**

    - Mở mục Phân tích CV.
    - Tải lên PDF, JPG hoặc PNG.
    - Nhấn Phân tích CV để xem kết quả.

    **Bước 2 — So sánh vị trí**

    - Mở mục So sánh vị trí.
    - Cung cấp CV và mô tả công việc.
    - Xem độ tương đồng văn bản và các từ khóa kỹ năng.

    **Bước 3 — Cải thiện CV**

    - Mở Gợi ý viết lại CV.
    - Xem các mục cần bổ sung và mẫu diễn đạt.
    - Chỉ sử dụng thông tin đúng với kinh nghiệm thực tế.

    **Bước 4 — Luyện phỏng vấn**

    - Nhập vị trí ứng tuyển và mô tả công việc.
    - Tạo câu hỏi và ghi lại ý chính cho câu trả lời.

    **Lưu ý về dữ liệu**

    - CV có thông tin cá nhân nhạy cảm; chỉ tải lên tài liệu bạn có quyền sử dụng.
    - Kết quả tự động có thể sai hoặc bỏ sót nội dung.
    - Điểm checklist và độ tương đồng chỉ mang tính tham khảo.
    """)


# =========================================================
# 13. GIỚI THIỆU
# =========================================================
elif page == "Giới thiệu":
    page_header(
        "Giới thiệu CVision AI",
        "Công cụ hỗ trợ người dùng chuẩn bị hồ sơ ứng tuyển."
    )

    st.markdown("""
    ### Mục tiêu

    CVision AI hỗ trợ người dùng kiểm tra cấu trúc CV, nhận diện từ khóa
    kỹ năng, đối chiếu nội dung với mô tả công việc và chuẩn bị phỏng vấn.

    ### Công nghệ sử dụng

    - **Python:** xử lý dữ liệu và logic ứng dụng.
    - **Streamlit:** xây dựng giao diện web.
    - **PyMuPDF:** trích xuất văn bản từ PDF.
    - **Pillow và pytesseract:** hỗ trợ xử lý ảnh và OCR khi môi trường
      đã được cấu hình phù hợp.
    - **TF-IDF và Cosine Similarity:** tính độ tương đồng văn bản.

    ### Giới hạn

    Hệ thống không thay thế chuyên gia tuyển dụng. Kết quả phụ thuộc
    vào chất lượng CV, văn bản trích xuất và các từ khóa được hỗ trợ.
    """)

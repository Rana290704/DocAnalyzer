import os
import streamlit as st
import openai
from io import BytesIO
import fitz  # PyMuPDF
from docx import Document
from PIL import Image
import pytesseract
import re

# --- Page Configuration ---
st.set_page_config(page_title="Legal Document Analyzer", layout="wide", initial_sidebar_state="collapsed")

# --- API Key Setup ---
# This setup is for OLDER openai library versions (below v1.0)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it as an environment variable or in Streamlit secrets.", icon="🚨")
    st.stop()

# Set the key globally for the old library
openai.api_key = OPENAI_API_KEY


# --- Text Extraction Functions (No changes here) ---
def extract_pdf_text(uploaded_file):
    # ... (code is identical)
    file_bytes = uploaded_file.read()
    text = ""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text")
    except Exception as e:
        st.warning(f"Could not read PDF directly: {e}. Falling back to OCR.")
        text = ""
    return text, file_bytes

def ocr_pdf_bytes(file_bytes):
    # ... (code is identical)
    text = ""
    with st.spinner("Performing OCR on PDF... this may take a moment."):
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                for page_num, page in enumerate(pdf):
                    st.text(f"Processing page {page_num + 1}/{len(pdf)}...")
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text += pytesseract.image_to_string(img) + "\n"
        except Exception as e:
            st.error(f"OCR failed: {e}")
    return text

def extract_docx_text(uploaded_file):
    # ... (code is identical)
    file_stream = BytesIO(uploaded_file.read())
    doc = Document(file_stream)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_image_text(uploaded_file):
    # ... (code is identical)
    with st.spinner("Performing OCR on image..."):
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)

# --- OpenAI Functions (MODIFIED FOR OLD LIBRARY) ---
def classify_document(text):
    prompt = f"""
    Please classify the following document into one of these categories:
    1. Housing Agreement
    2. Employment Contract
    3. Lease Agreement
    4. Business Contract
    5. Service Agreement
    6. Non-Disclosure Agreement (NDA)
    7. Other (Please specify)

    Document Preview:
    {text[:1500]}
    """
    try:
        # OLD SYNTAX
        resp = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a legal document classification assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.1
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return "Classification Failed"

def analyze_document_with_openai(text, doc_type):
    prompt = f"""
    You are a meticulous legal analyst specializing in Indian contract law.
    Analyze the following '{doc_type}' document. Provide a clear, structured breakdown.

    Follow this format EXACTLY. Include ALL headings, even if the section is empty (write "None found." in that case).

    ### Overall Analysis
    Provide a brief, high-level summary of the document's purpose and quality.

    ### Ambiguous Clauses
    - List any clauses that are vague, unclear, or open to multiple interpretations. Quote the clause and explain the ambiguity.

    ### Missing Terms
    - Identify any crucial legal terms or clauses that are missing (e.g., Termination, Confidentiality, Dispute Resolution, Governing Law).

    ### Red Flags 🚩
    - Quote the exact text for any critical issues, high-risk items, or clauses that are unfair or potentially non-compliant with Indian law. Explain precisely why each is a major problem.

    ### Green Flags ✅
    - Quote any clauses that are well-drafted, fair, and protective of the parties' interests. Explain why they are good.

    ### Suggestions for Improvement
    - Provide a numbered list of actionable suggestions to improve the document, making it more robust and clear.

    ---
    DOCUMENT TEXT:
    {text[:4000]}
    """
    try:
        # OLD SYNTAX
        resp = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=2500,
            temperature=0.4
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return "Analysis Failed"

# --- Helper Function for Parsing (No changes here) ---
def extract_section(full_text, heading):
    # ... (code is identical)
    pattern = re.compile(f"### {re.escape(heading)}(.*?)(?=\n### |$)", re.S | re.I)
    match = pattern.search(full_text)
    if match:
        content = match.group(1).strip()
        if content.lower() in ["", "none", "none found."]:
            return None
        return content
    return None

# --- Streamlit UI (No changes here) ---
st.title("⚖️ Legal Document Analyzer")
# ... (rest of the UI code is identical)
st.markdown("Upload a document (PDF, DOCX, or Image) to get an AI-powered legal analysis.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.write(f"**Filename:** `{uploaded_file.name}`")
    text, file_bytes = "", None
    mime_type = uploaded_file.type

    if mime_type == "application/pdf":
        text, file_bytes = extract_pdf_text(uploaded_file)
        if len(text.strip()) < 100 and file_bytes:
            st.info("Initial text extraction was minimal. Performing full OCR on the PDF.")
            text = ocr_pdf_bytes(file_bytes)
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_docx_text(uploaded_file)
    elif mime_type.startswith("image/"):
        text = extract_image_text(uploaded_file)

    if not text or len(text.strip()) < 20:
        st.error("Could not extract sufficient text from the document. It might be empty, corrupted, or a complex image-based file.", icon="📄")
    else:
        with st.expander("View Extracted Text"):
            st.text_area("", text, height=300)

        if st.button("Analyze Document", type="primary"):
            with st.spinner("Step 1/2: Classifying document..."):
                doc_type = classify_document(text)
            st.success(f"**Document Type Identified:** {doc_type}")

            with st.spinner("Step 2/2: Performing in-depth analysis..."):
                analysis_text = analyze_document_with_openai(text, doc_type)

            if analysis_text == "Analysis Failed":
                st.stop()

            st.subheader("📝 Analysis Breakdown")
            red_flags = extract_section(analysis_text, "Red Flags 🚩")
            green_flags = extract_section(analysis_text, "Green Flags ✅")
            overall_analysis = extract_section(analysis_text, "Overall Analysis")
            ambiguous_clauses = extract_section(analysis_text, "Ambiguous Clauses")
            missing_terms = extract_section(analysis_text, "Missing Terms")
            suggestions = extract_section(analysis_text, "Suggestions for Improvement")

            if overall_analysis:
                st.markdown("#### Overall Analysis")
                st.markdown(overall_analysis)

            if red_flags:
                st.markdown("#### Red Flags 🚩")
                st.error(red_flags)
            if green_flags:
                st.markdown("#### Green Flags ✅")
                st.success(green_flags)
            if not red_flags and not green_flags:
                st.info("No specific Red or Green Flags were identified in the analysis.", icon="ℹ️")
            if ambiguous_clauses:
                st.markdown("#### Ambiguous Clauses")
                st.warning(ambiguous_clauses)
            if missing_terms:
                st.markdown("#### Missing Terms")
                st.warning(missing_terms)
            if suggestions:
                st.markdown("#### Suggestions for Improvement")
                st.info(suggestions)
            with st.expander("View Full Raw AI Response"):
                st.text(analysis_text)
else:
    st.info("Please upload a file to begin the analysis.")

import os
import streamlit as st
import openai
from io import BytesIO
import fitz  # PyMuPDF
from docx import Document
from PIL import Image
import pytesseract

# Page config (optional)
st.set_page_config(page_title="Legal Document Analyzer", layout="wide")

# Load API key from environment or Streamlit secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or in Streamlit secrets.")
    st.stop()
openai.api_key = OPENAI_API_KEY

# PDF text extraction
def extract_pdf_text(uploaded_file):
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text, file_bytes

# OCR fallback for scanned PDFs
def ocr_pdf_bytes(file_bytes):
    text = ""
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    for page in pdf:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img.convert("L"))
    return text

# DOCX extraction
def extract_docx_text(uploaded_file):
    file_stream = BytesIO(uploaded_file.read())
    doc = Document(file_stream)
    text = "\n".join(p.text for p in doc.paragraphs)
    return text, None

# Image OCR
def extract_image_text(uploaded_file):
    image = Image.open(uploaded_file)
    return pytesseract.image_to_string(image.convert("L"))

# Classification via OpenAI
def classify_document(text):
    prompt = f"""
    Please classify the following document into one of these categories:
    1. Housing Agreement
    2. Employment Contract
    3. Lease Agreement
    4. Business Contract
    5. Tax Invoice
    6. Bank Check
    7. Other (Specify)

    Document:
    {text[:1000]}
    """
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":"You are a legal expert familiar with Indian law."},
            {"role":"user","content":prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()

# Analysis via OpenAI, including Red Flags
def analyze_document_with_openai(text, doc_type):
    prompt = f"""
    Analyze the following {doc_type} for ambiguous clauses, missing terms, and non-compliance with Indian law.
    Provide:
    1. Ambiguous Clauses
    2. Missing Terms
    3. Potential Non-Compliance
    4. Suggestions
    5. Red Flags (critical issues that must be fixed before submission)

    Document:
    {text[:2000]}
    """
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":"You are a legal expert trained in Indian law."},
            {"role":"user","content":prompt}
        ],
        max_tokens=2000,
        temperature=0.5
    )
    return resp.choices[0].message.content.strip()

# Streamlit UI
st.title("Legal Document Analyzer")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or image", type=["pdf","docx","png","jpg","jpeg"])
if not uploaded_file:
    st.info("Please upload a file to get started.")
else:
    st.write(f"**Filename:** {uploaded_file.name}")
    text, file_bytes = "", None

    mime = uploaded_file.type
    if mime == "application/pdf":
        text, file_bytes = extract_pdf_text(uploaded_file)
        if len(text.strip()) < 50 and file_bytes:
            st.info("Performing OCR on PDF pages…")
            text = ocr_pdf_bytes(file_bytes)

    elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text, _ = extract_docx_text(uploaded_file)

    elif mime.startswith("image/"):
        st.info("Extracting text via OCR from image…")
        text = extract_image_text(uploaded_file)

    else:
        st.error("Unsupported file type.")
        st.stop()

    st.subheader("Extracted Text Preview")
    st.text_area("", text[:1500], height=300)

    if st.button("Classify & Analyze"):
        with st.spinner("Classifying…"):
            doc_type = classify_document(text)
        st.success(f"Document Type: **{doc_type}**")

        with st.spinner("Analyzing…"):
            analysis = analyze_document_with_openai(text, doc_type)

        # Split out Red Flags section if present
        red_flags = None
        main_analysis = analysis
        if "Red Flags" in analysis:
            parts = analysis.split("Red Flags")
            # parts[0] ends before the keyword, parts[1] contains the section
            main_analysis = parts[0].strip()
            red_flags = parts[1].strip(': \n')

        st.subheader("Analysis")
        st.write(main_analysis)

        if red_flags:
            st.subheader("Red Flags")
            st.error(red_flags)

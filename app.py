import streamlit as st
import fitz
from transformers import pipeline, MBart50TokenizerFast, MBartForConditionalGeneration
from multiprocessing import Pool, cpu_count
import tempfile

# Load summarization pipeline
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

# Load translation model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

# Define max chunk length
max_chunk_length = 1024

# Function to chunk text
def chunk_text(text, max_chunk_length):
    chunks = []
    current_chunk = ""
    for sentence in text.split("."):
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
            if current_chunk != "":
                current_chunk += " "
            current_chunk += sentence.strip()
        else:
            chunks.append(current_chunk)
            current_chunk = sentence.strip()
    if current_chunk != "":
        chunks.append(current_chunk)
    return chunks

# Function to summarize and translate a chunk
def summarize_and_translate_chunk(chunk, lang):
    summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
    summary_text = summary[0]['summary_text']

    # Translate summary
    translated_chunk = translate_summary(summary_text, lang)
    return translated_chunk

# Function to translate the summary
def translate_summary(summary, lang):
    # Chunk text if it exceeds maximum length
    if len(summary) > max_chunk_length:
        chunks = chunk_text(summary, max_chunk_length)
    else:
        chunks = [summary]

    # Translate each chunk
    translated_chunks = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[lang],
            max_length=1024,
            num_beams=4,
            early_stopping=True,
            length_penalty=2.0,
        )
        translated_chunks.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])

    return " ".join(translated_chunks)



# Function to read PDF and summarize and translate chunk by chunk
def summarize_and_translate_pdf(uploaded_file, lang):
    # Save uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        doc = fitz.open(temp_file_path)
    except FileNotFoundError:
        st.error("File not found. Please make sure the file path is correct.")
        return []

    total_chunks = len(doc)
    chunks = []

    for i in range(total_chunks):
        page = doc.load_page(i)
        text = page.get_text()
        chunks.extend([text[j:j+max_chunk_length] for j in range(0, len(text), max_chunk_length)])

    # Use multiprocessing to parallelize the process
    with Pool(cpu_count()) as pool:
        translated_chunks = pool.starmap(summarize_and_translate_chunk, [(chunk, lang) for chunk in chunks])

    # Delete temporary file
    os.unlink(temp_file_path)

    return translated_chunks


# Streamlit UI
st.title("PDF Summarization and Translation")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    # Display uploaded file
    st.write("Uploaded PDF file:", uploaded_file.name)

    # Language selection
    languages = {
        "Arabic": "ar_AR", "Czech": "cs_CZ", "German": "de_DE", "English": "en_XX", "Spanish": "es_XX",
        "Estonian": "et_EE", "Finnish": "fi_FI", "French": "fr_XX", "Gujarati": "gu_IN", "Hindi": "hi_IN",
        "Italian": "it_IT", "Japanese": "ja_XX", "Kazakh": "kk_KZ", "Korean": "ko_KR", "Lithuanian": "lt_LT",
        "Latvian": "lv_LV", "Burmese": "my_MM", "Nepali": "ne_NP", "Dutch": "nl_XX", "Romanian": "ro_RO",
        "Russian": "ru_RU", "Sinhala": "si_LK", "Turkish": "tr_TR", "Vietnamese": "vi_VN", "Chinese": "zh_CN",
        "Afrikaans": "af_ZA", "Azerbaijani": "az_AZ", "Bengali": "bn_IN", "Persian": "fa_IR", "Hebrew": "he_IL",
        "Croatian": "hr_HR", "Indonesian": "id_ID", "Georgian": "ka_GE", "Khmer": "km_KH", "Macedonian": "mk_MK",
        "Malayalam": "ml_IN", "Mongolian": "mn_MN", "Marathi": "mr_IN", "Polish": "pl_PL", "Pashto": "ps_AF",
        "Portuguese": "pt_XX", "Swedish": "sv_SE", "Swahili": "sw_KE", "Tamil": "ta_IN", "Telugu": "te_IN",
        "Thai": "th_TH", "Tagalog": "tl_XX", "Ukrainian": "uk_UA", "Urdu": "ur_PK", "Xhosa": "xh_ZA",
        "Galician": "gl_ES", "Slovene": "sl_SI"
    }

    lang = st.selectbox("Select language for translation", list(languages.keys()))

    # Translate PDF
    if st.button("Summarize and Translate"):
        translated_chunks = summarize_and_translate_pdf(uploaded_file, languages[lang])
        
        # Display translated text
        st.header("Translated Summary")
        for chunk in translated_chunks:
            st.write(chunk)

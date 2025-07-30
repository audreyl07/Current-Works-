import warnings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import importlib
from fpdf import FPDF

# Suppress encoder_attention_mask warning from transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

# --- PDF Text Extraction ---
def extract(file_path):
    """
    Extracts and splits text from a PDF file into manageable chunks.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(pages)
    return texts

# --- Summarization with Clustering ---
def summarize_document_with_kmeans_clustering(file, llm, embeddings):
    """
    Clusters document chunks and summarizes them using an LLM.
    Includes code and syntax examples in the summary.
    """
    filter = EmbeddingsClusteringFilter(embeddings=embeddings, num_clusters=10)
    texts = extract(file)
    try:
        filtered_docs = filter.transform_documents(documents=texts)

        # Custom prompt to include code and syntax examples
        custom_prompt = PromptTemplate.from_template(
            "Summarize the following document. Include any code examples, and format them clearly:\n\n{text}"
        )
        summarization_chain = load_summarize_chain(
            llm, chain_type="stuff", prompt=custom_prompt
        )

        result = summarization_chain.invoke({"input_documents": filtered_docs})

        # Return the summary text
        if isinstance(result, dict):
            return result.get("output_text", str(result))
        else:
            return str(result)

    except Exception as e:
        return f"[ERROR]: {e}"

# --- Append Summary to PDF ---
def append_summary_to_pdf(pdf_path, new_summary, output_pdf="summary_output.pdf"):
    """
    Appends a new summary to an existing PDF summary file.
    If the file exists, previous summaries are included.
    """
    from fpdf import FPDF
    import os

    # Extract existing text from the PDF if it exists
    existing_text = ""
    if os.path.exists(pdf_path):
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        existing_text += page_text + "\n"
        except ImportError:
            print("pdfplumber not installed, skipping existing PDF text extraction.")

    # Create a new PDF and add previous and new summaries
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Times", size=12)

    if existing_text:
        pdf.multi_cell(0, 10, "=== Previous Summary ===")
        for line in existing_text.split('\n'):
            pdf.multi_cell(0, 10, line)
        pdf.ln(10)

    pdf.multi_cell(0, 10, "=== New Summary ===")
    for line in new_summary.split('\n'):
        pdf.multi_cell(0, 10, line)

    pdf.output(output_pdf)
    print(f"Combined summary saved to {output_pdf}")

# --- Model and Embedding Configuration ---
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)

llm = OllamaLLM(
    model="llama3.1:8b",
    temperature=0
)

# --- Summarize First PDF ---
summary1 = summarize_document_with_kmeans_clustering(
    "d:/work/coursepdfproject/COMP2401_Ch1_SystemsProgramming.pdf",
    llm,
    embeddings
)

print("\n===== PDF SUMMARY 1 =====\n")
print(summary1)
print("\n=========================\n")

# --- Save First Summary to PDF ---
pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font("Times", size=12)
for line in summary1.split('\n'):
    pdf.multi_cell(0, 10, line)
pdf.output("summary_output.pdf")
print("Summary saved to summary_output.pdf")

# --- Prompt User for New PDF Path ---
new_pdf_path = input("Enter the path to the new PDF to summarize and append: ").strip()

# --- Summarize User-Provided PDF ---
summary2 = summarize_document_with_kmeans_clustering(
    new_pdf_path,
    llm,
    embeddings
)

print("\n===== PDF SUMMARY 2 =====\n")
print(summary2)
print("\n=========================\n")

# --- Append New Summary to Existing PDF ---
append_summary_to_pdf("summary_output.pdf", summary2, output_pdf="summary_output_combined.pdf")

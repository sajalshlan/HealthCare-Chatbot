import argparse
import os
import re
import sys
import warnings
import pdfplumber
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
EMBED_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_AUTH_TOKEN = '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'

def setup_device() -> str:
    return f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

def initialize_embedding_model(device: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )

def initialize_llm(device: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_config = AutoConfig.from_pretrained(LLM_MODEL_ID, use_auth_token=HF_AUTH_TOKEN)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=HF_AUTH_TOKEN
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, use_auth_token=HF_AUTH_TOKEN)

    generate_text = pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        max_new_tokens=3000,  # Increased to 3000
        repetition_penalty=1.1
    )

    return HuggingFacePipeline(pipeline=generate_text)

def extract_text_from_pdf(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_with_ocr(file_path: str) -> str:
    os.environ['USE_TORCH'] = '1'
    predictor = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_pdf(file_path) if file_path.endswith('.pdf') else DocumentFile.from_images(file_path)
    result = predictor(doc)
    return result.render()

def extract_text_from_website(url: str) -> str:
    loader = WebBaseLoader(url)
    data = loader.load()
    return "\n".join(doc.page_content for doc in data)

def extract_file_data(file_path_or_url: str) -> str:
    if file_path_or_url.startswith(('http://', 'https://')):
        return extract_text_from_website(file_path_or_url)
    elif file_path_or_url.endswith('.pdf'):
        with pdfplumber.open(file_path_or_url) as pdf:
            if pdf.pages and len(pdf.pages[0].extract_text() or "") > 50:
                return extract_text_from_pdf(file_path_or_url)
    return extract_text_with_ocr(file_path_or_url)

def create_vector_store(texts, embed_model):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
    documents = text_splitter.create_documents([texts])
    return FAISS.from_documents(documents, embedding=embed_model)

def extract_helpful_answer(text: str) -> str:
    pattern = re.compile(r"Helpful Answer:(.*)", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else "Helpful Answer not found"

def process_query(rag_pipeline, query: str, extracted_text: str) -> str:
    prompt = f"""
    <<SYS>>
    You are an advanced healthcare informatory AI assistant designed to extract and provide key information from various healthcare documents. Your primary functions are to:

    1. Extract relevant healthcare information from provided documents such as insurance policies, hospital forms, medical bills, discharge summaries, and prescription information sheets.
    2. Organize the extracted information into clear, structured categories.
    3. Answer user queries based on the extracted information with accuracy and relevance.
    4. Maintain patient confidentiality and adhere to healthcare information guidelines.
    5. Provide concise yet comprehensive responses, offering to elaborate when necessary.

    Always prioritize accuracy in your extractions and responses. If any information is unclear or unavailable, state this explicitly rather than making assumptions.
    <</SYS>>

    [INST]
    Based on the healthcare documents provided, please extract and organize the following key information:

    1. Insurance and Coverage:
       - Policy details (limits, deductibles, co-pays)
       - Coverage exclusions and limitations
       - Eligibility criteria for specific treatments or services

    2. Medical Procedures and Treatments:
       - Treatment plans and protocols
       - Follow-up instructions
       - Prescription drug information (dosages, side effects, interactions)

    3. Financial and Administrative:
       - Claim procedures and required documentation
       - Billing information (codes, charges, payment options)
       - Contact information for customer support or care providers

    4. Patient Information:
       - Rights and responsibilities
       - Emergency procedures and contact information
       - Consent forms and privacy policies

    After extracting this information, be prepared to:
    1. Provide structured summaries of each category.
    2. Answer specific user queries about any of these topics.
    3. Offer relevant cross-references between different information points when applicable.
    4. Highlight any critical information that requires immediate attention or action.

    For each query, provide a concise yet informative answer. If the information isn't available in the provided documents or requires medical expertise beyond the scope of these documents, clearly state this limitation.
    [/INST]

    [INST]
    {query}

    Extracted Text:
    {extracted_text}
    [/INST]
    """

    answer = rag_pipeline(prompt)['result']
    return extract_helpful_answer(answer)

def main():
    parser = argparse.ArgumentParser(description="Answers to your questions on the information you provide.")
    parser.add_argument('input_source', type=str, help='Path to the file (PDF or image) or URL')
    args = parser.parse_args()

    try:
        device = setup_device()
        embed_model = initialize_embedding_model(device)
        llm = initialize_llm(device)

        extracted_text = extract_file_data(args.input_source)
        print("Text extracted successfully.")

        vector_store = create_vector_store(extracted_text, embed_model)
        rag_pipeline = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vector_store.as_retriever())

        while True:
            query = input("Enter your query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            answer = process_query(rag_pipeline, query, extracted_text)
            print("\nAnswer:", answer)
            print("\n" + "-"*50 + "\n")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
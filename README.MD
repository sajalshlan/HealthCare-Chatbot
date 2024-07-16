Healthcare Informatory AI Chatbot
==================================
Author - Sajal Sharma (@sajalshlan)

Problem Statement
-----------------
In the healthcare industry, extracting and interpreting information from various documents (such as insurance policies, medical records, and treatment plans) can be time-consuming and prone to human error. This project aims to solve this problem by developing an AI-powered tool that can:

1. Automatically extract key information from healthcare documents in various formats (HTML/websites and PDFs).
2. Provide accurate and relevant answers to user queries based on the extracted information.
3. Handle both digital and scanned documents, ensuring broad applicability across different document types.
4. Offer a user-friendly interface for healthcare professionals and patients to quickly access the information they need.

Project Overview
----------------
This Healthcare Informatory AI Chatbot is a locally deployable information extraction tool designed for healthcare organizations. It leverages advanced natural language processing techniques to parse complex healthcare documents and provide instant, accurate responses to user queries.

Key Features
------------
1. Multi-format Document Processing: 
   - Extracts text from both HTML websites and PDF documents.
   - Handles native digital PDFs and scanned documents(scanned images as well as scanned pdfs) requiring OCR.

2. Advanced Language Model Integration:
   - Utilizes the open-source Mistral-7B-Instruct-v0.3 LLM for sophisticated query processing.
   - Provides context-aware responses based on the extracted document information.

3. Intelligent Information Extraction:
   - Focuses on key healthcare-related information points such as insurance details, medical procedures, financial information, and patient rights.
   - Organizes extracted information into clear, structured categories for easy retrieval.

4. User-Friendly Interface:
   - Offers a straightforward command-line interface for user queries.
   - Supports both local execution and Google Colab deployment for flexibility.

Installation
------------
Prerequisites:
- Python 3.7 or higher
- Git
- Pip (Python package installer)
- (Optional) CUDA-capable GPU for faster processing

If you prefer the fast-run way, go for the colab process.

Local Installation:
1. Clone the repository:

   ```git clone https://github.com/your-username/healthcare-informatory-ai-chatbot.git```
   ```cd healthcare-informatory-ai-chatbot```

2. Create and activate a virtual environment:

   ```python3 -m venv healthcare_env```
   ```source healthcare_env/bin/activate  # On Windows, use "`healthcare_env\Scripts\activate`"```

3. Install PyTorch (visit https://pytorch.org/ for the command specific to your system):

   ```pip3 install torch torchvision torchaudio```

4. Install Swig (required for some dependencies):
   - On macOS: ```brew install swig```
   - On Ubuntu: ```sudo apt-get install swig```
   - On Windows: Download from http://www.swig.org/download.html and add to PATH

5. Install setuptools:

   ```pip install setuptools```

6. Install the required packages:

   ```pip3 install -r requirements.txt```

Google Colab Installation:
1. Open a new Google Colab notebook.
2. Upload script.py and any documents you want to analyze to your Colab environment.
3. Run the following command in a code cell to install dependencies:
```
   !pip install -qU \
   transformers \
   sentence-transformers==2.2.2 \
   datasets==2.14.0 \
   accelerate==0.21.0 \
   einops==0.6.1 \
   langchain \
   bitsandbytes>=0.41.3 \
   faiss-cpu==1.7.4 \
   langchain-community \
   python-doctr[torch] \
   pdfplumber \
   tf2onnx
```
Usage
-----
Local Usage:
1. Activate your virtual environment if not already active:

   ```source healthcare_env/bin/activate  # On Windows, use "`healthcare_env\Scripts\activate`"```

2. Run the script with your document or URL:

  ```python3 script.py path/to/your/document.pdf```
   or
   ```python3 script.py https://www.example.com/healthcare-document```

3. Once the document is processed, you can enter queries at the prompt.

Google Colab Usage:
1. After installation, run the script in a code cell:

   ```!python3 script.py "/content/your_document.pdf"```
   or
   ```!python3 script.py "https://www.example.com/healthcare-document"```

2. Enter your queries in the output area when prompted.

Example Queries
---------------
- "What are the coverage limits for outpatient procedures?"
- "Explain the process for filing a claim after an emergency room visit."
- "What are the patient's rights regarding access to medical records?"
- "Detail the prescription drug coverage, including any restrictions or prior authorization requirements."

Customization
-------------
- To add new types of information extraction, modify the `process_query` function in script.py.
- To use a different language model, update the LLM_MODEL_ID constant and adjust the model initialization in the initialize_llm function.

Limitations
-----------
- The accuracy of responses depends on the quality and completeness of the input documents.
- Very large documents may require significant processing time and memory.
- The tool does not have real-time access to external databases or up-to-date healthcare information beyond what's in the provided documents.


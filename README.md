
## Project Overview

This project implements a local Multimodal Retrieval-Augmented Generation (MRAG) system designed to process heterogeneous data sources including academic PDFs, images, and source code. Unlike standard RAG systems that flatten all data into text, this system utilizes a Late-Fusion Architecture. This approach processes each modality (visual, textual, and code) through specialized encoding streams before fusing the retrieved information for the Large Language Model.

The system is designed for local deployment on Windows.

## System Workflow

The architecture follows a strict pipeline to ensure high-precision retrieval:

1.  **Ingestion:** Files are routed to specific parsers based on type. PDFs are parsed for structure, images are captioned using BLIP, and code is segmented by logical blocks.
2.  **Vectorization:** Data is converted into high-dimensional vectors. We use BGE-M3 for text and code, and CLIP for visual alignment.
3.  **Storage:** Vectors are stored in a Qdrant database, segregated into specific collections (`mrag_pdf`, `mrag_image`, `mrag_code`).
4.  **Hybrid Retrieval:** User queries trigger parallel searches across all collections.
5.  **Re-Ranking:** A Cross-Encoder layer evaluates the raw search results to filter out irrelevant chunks before they reach the LLM.
6.  **Generation:** The Qwen2.5 model synthesizes the final answer using the verified context.

## File Descriptions

### app.py
The main entry point built with Flask. It defines API endpoints for file ingestion and user queries, managing communication between the UI and the backend pipeline.

### mrag_pipeline.py
Contains the core Late-Fusion logic. It manages the Qdrant connection, initializes embedding models (BGE-M3, CLIP), and processes file types independently to preserve modality-specific features.

### llm_interface.py
Handles interaction with the local LLM via Ollama. It constructs prompt contexts from retrieved documents and performs the re-ranking step to reduce hallucinations.

### validate_system.py
An automated regression testing script. It runs a "Golden Dataset" of questions against the system to verify accuracy across Architecture, Vision, and Code modalities.

### index.html
The user interface providing a dashboard for file uploads and a chat interface. It features a Transparent Retrieval view to inspect raw evidence and relevance scores.

## Tools and Technologies Used

* **Language:** Python 3.10+
* **Web Framework:** Flask
* **Vector Database:** Qdrant (Local Docker instance)
* **LLM Serving:** Ollama
* **Embedding Models:** BGE-M3 (Text/Code), CLIP (Vision)
* **Vision Models:** BLIP (Image Captioning)
* **PDF Parsing:** Unstructured, PyMuPDF
* **Validation:** Python Requests module

## Setup and Installation Guide (Windows)

Follow these steps to set up the project on a Windows machine.

### Prerequisites
* **Python 3.10+** installed and added to PATH.
* **Docker Desktop** installed and running.
* **Ollama** installed and running.

### Step 1: Clone the Repository
Extract the project files into a folder named `mrag-project`. Open PowerShell and navigate to this folder.

```powershell
cd mrag-project 
```

### Step 2: Set up a Virtual Environment
Create and activate a virtual environment to manage dependencies.

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Required Packages
Install the necessary Python libraries.

```powershell
pip install flask flask-cors qdrant-client sentence-transformers transformers torch pillow pymupdf unstructured rank_bm25 requests
```

### Step 4: Configure the Local LLM
Ensure Ollama is running in the system tray, then pull the required model.

```powershell
ollama pull qwen2.5:7b
```

### Step 5: Start the Vector Database (Qdrant)
Run Qdrant using Docker Desktop.

```powershell
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant-mrag qdrant/qdrant
```

### Step 6: Add Test Data
Since data files are not included in this repository to maintain size limits:

* Create a folder named data in the project root
* Open index.html in your browser.
* Upload these files via the web interface once the app is running.

### Step 7: Run the Application
Start the Flask server.
```powershell
python app.py
```
Access the interface at: http://localhost:5000

### Step 8: Open Frontend
Start chatting with MRAG and upload files.

### Step 9: Validation
To verify the system is working, open a new PowerShell window, activate the environment, and run the validation script.
```powershell
.\venv\Scripts\activate
python validate_system.py
```

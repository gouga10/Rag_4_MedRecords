
# Health-RAG: A Simple RAG Application for Electronic Medical Records

Health-RAG is a Retrieval-Augmented Generation (RAG) application designed to answer questions about patients using Electronic Medical Records (EMRs).

## Dataset & Approach

The original dataset contained **55,000 samples**. To minimize costs,compute and have abetter performing app two smaller subsets were used:
- **Scenario 1:** 100 samples
- **Scenario 2:** 1,000 samples

## Implementation Steps

### 1. Data Preprocessing
- Identified and handled missing data
- Identified duplicate entries
- Converted all text to lowercase for consistency

Since Large Language Models (LLMs) perform better with structured text, each row in the CSV file was transformed into a full sentence that better represents the data in context. The processed data was then saved as a single PDF.

### 2. Creating Vector Databases
Two separate vector databases were created for each sample size (100 & 1,000):
- **One for the PDF version**
- **One for the CSV version**

The embeddings were generated using OpenAI’s `text-embedding-3-large` model.

### 3. Retrieval & Ranking
- **Retrieval:** LLamaIndex was used to fetch the top 4 most relevant chunks based on the query.
- **Reranking:** A reranker was applied to refine the retrieved chunks and prioritize the most relevant information.

### 4. Response Generation
- The chat engine generates responses using the retrieved context.
- The entire conversation history is passed to both the retriever and the response prompt to ensure continuity and awareness of the conversation’s state.

### 5. API & UI Integration
- The application is served using **FastAPI**.
- A **Streamlit** frontend allows users to interact with the system.

## How to Use

### 1. Set up the Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API Key
- Add your OpenAI API key to the `.env` file.

### 4. Run the Application

-navigate to Health-RAG 
-
```bash
# Start the Streamlit UI
streamlit run streamlit.py
```
```bash
# In a separate terminal, start the API
cd Health-RAG/RAG
python api.py
```

### 5. Interact with the App
- Open your browser and start chatting with the application.

---

This README provides a clear and structured guide for setting up and running Health-RAG. 🚀


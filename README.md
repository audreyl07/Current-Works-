# Current-Works-

## Course Notes PDF Summarizer 
This project works as a way to summarize textbook chapters into short, concise summaries that are easy to read, have examples and are more understandable than a block of text. Also no one has time to be reading a textbook when you can just make something to do the hard reading for you. 

### Features 
- Extracts and splits texts from PDFs
- Uses the K-means clustering algorithm to help improve relevance and speed
- Employs an LLM (llama3.1:8b), via Ollama to summarize the text
- Highlights code examples in the summaries
- Can combine the old pdf reports with new summaries

### Installation 
#### Requirements 
- Python 3.8+
- install the dependencies via pip
- have Ollama installed locally 

```bash
pip install langchain langchain-community langchain-core langchain-ollama langchain-huggingface sentence-transformers fpdf pdfplumber torch
```

### How it Works 
1. extracts the PDF into text
2. splits the text into chunks
3. clusters similar chunks using the k-means clustering algorithm
4. summarizes using LLM
5. exports summary into a PDF
6. allows appending new summaries into the previous PDF

### How to Use 
1. given a pdf or you can use your own (make sure to change code if you do), run the program first
2. you will then be prompted to provide the path to a second PDF
3. this will take the previous summary pdf and append the new second summary to it
   




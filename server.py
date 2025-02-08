from flask import Flask, request, jsonify
import re
import cv2
import json
import ollama
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration


app = Flask(__name__)
llava_model_name = "llava-hf/llava-1.5-7b-hf"
llava = LlavaForConditionalGeneration.from_pretrained(llava_model_name)
llava_processor = AutoProcessor.from_pretrained(llava_model_name)


def summarize_page(page_text, page_number):
    prompt = f"""
You are an expert academic summarizer. Summarize the following page from a research paper in 100-150 words, keeping key findings and figure insights.

### Page {page_number}:
{page_text}

**Summary:**
    """
    ollama.pull('llama3.2')
    response: ollama.ChatResponse = ollama.chat(model='llama3.2', messages=[{
        'role': 'user',
        'content': prompt,
    }])
    return response.message.content


def summarize_figures(figures, model, processor):
    prompt = "USER: <image>\nSummarize this figure in 50 words. ASSISTANT:"
    summaries = ""
    for figure in figures:
        inputs = processor(images=figure, text=prompt, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=100)
        summary = llava_processor.batch_decode(
            output, skip_special_tokens=True)[0][54:]
        summaries += summary + "\n"
    return summaries


def get_page_figures(files, page):
    figures = []
    for file in files:
        match = re.match(rf"^{page}_[0-9]+\.jpg$", file.filename)
        if match:
            image_bytes = file.read()
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            figure = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            figures.append(figure)
    return figures


def get_full_paper_summary(files, full_text):
    full_paper_summary = ""
    for page in tqdm(range(len(full_text))):
        page_figures = get_page_figures(files, page)
        figure_summary = summarize_figures(page_figures, llava, llava_processor)
        page_text = full_text[page]
        page_text += "Figures: \n" + figure_summary
        page_summary = summarize_page(page_text, page)
        full_paper_summary += page_summary
    return full_paper_summary


def summarize_paper(full_summary_text):
    prompt = f"""
You are an advanced AI model capable of understanding and summarizing complex research papers. 
Your task is to generate a one-page summary that captures the main contributions, findings, 
and key insights of the entire paper.

### **Input:**
I will provide you with a set of page-wise summaries of a research paper. These summaries highlight 
key information extracted from each page, including textual content, figures, and tables.

### **Your Task:**
1. **Synthesize the information across all pages** into a well-structured, concise summary.  
2. **Capture the main objective** of the paper, its methodology, findings, and conclusions.  
3. **Maintain clarity and coherence**, ensuring the summary reads naturally as a single unit.  
4. **Prioritize key takeaways** over minor details.  

### **Output Format:**
- **Title of the Paper**  
- **Problem Statement** (What problem does the paper address?)  
- **Methodology** (How was the problem tackled?)  
- **Key Findings & Results** (What are the major insights?)  
- **Conclusion & Implications** (Why is this research important?)  

### **Input Data:**
{full_summary_text}

### **Output:**
Provide a **concise, well-structured one-page summary** of the entire research paper.
"""
    ollama.pull('llama3.2')
    response = ollama.chat(model='llama3.2', messages=[
                           {"role": "user", "content": prompt}])

    return response.message.content


@app.route('/summarize', methods=['POST'])
def summarize():
    files = request.files.getlist('figures')
    full_text = json.loads(request.form.get("full_text", {}))

    full_paper_summary = get_full_paper_summary(files, full_text)
    print("Final summary...")
    final_summary = summarize_paper(full_paper_summary)
    return jsonify(
        {
            "summary": final_summary
        }
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)

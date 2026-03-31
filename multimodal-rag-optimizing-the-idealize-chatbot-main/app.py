import os
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from mrag_pipeline import MRAGPipeline
from llm_interface import LLMInterface

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {
    'pdf': ['pdf'],
    'image': ['png', 'jpg', 'jpeg', 'gif', 'bmp'],
    'code': ['py', 'js', 'java', 'cpp', 'c', 'h', 'txt', 'md']
}

print("\n" + "="*50)
print("Initializing MRAG System Components")
print("="*50)

try:
    mrag = MRAGPipeline()
    llm = LLMInterface()
    print("System Ready")
except Exception as e:
    print(f"Initialization Error: {e}")
    raise e

def allowed_file(filename, file_type):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in ALLOWED_EXTENSIONS.get(file_type, [])


@app.route('/ingest/pdf', methods=['POST'])
def ingest_pdf():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename, 'pdf'): return jsonify({"error": "Invalid PDF"}), 400
    
    filename = secure_filename(file.filename)
    filepath = UPLOAD_FOLDER / filename
    file.save(filepath)
    
    try:
        result = mrag.ingest_pdf(str(filepath))
        return jsonify({"success": True, "filename": filename, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ingest/code', methods=['POST'])
def ingest_code():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename, 'code'): return jsonify({"error": "Invalid Code file"}), 400
    
    filename = secure_filename(file.filename)
    filepath = UPLOAD_FOLDER / filename
    file.save(filepath)
    
    try:
        result = mrag.ingest_code(str(filepath))
        return jsonify({"success": True, "filename": filename, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ingest/image', methods=['POST'])
def ingest_image():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename, 'image'): return jsonify({"error": "Invalid Image"}), 400
    
    filename = secure_filename(file.filename)
    filepath = UPLOAD_FOLDER / filename
    file.save(filepath)
    
    try:
        result = mrag.ingest_image(str(filepath)) 
        return jsonify({"success": True, "filename": filename, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('message') or data.get('query')
    
    if not query_text: return jsonify({"error": "No query"}), 400
    
    try:
        candidates = mrag.retrieve(query_text, top_k=5)
        
        result = llm.query(query_text, candidates)
        
        return jsonify({
            "success": True,
            "response": result['answer'],
            "answer": result['answer'],
            "sources": result['sources'],
            "evidence": result.get('evidence', []),
            "candidates_retrieved": result.get('candidates_retrieved', len(candidates)),
            "candidates_reranked": result.get('candidates_reranked', len(candidates)),
            "context_used": result.get('context_used', 5)
        })
    except Exception as e:
        print(f"Query Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify(mrag.get_stats())

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/clear', methods=['POST'])
def clear():
    try:
        for name in ['mrag_pdf', 'mrag_image', 'mrag_code']:
            try: mrag.client.delete_collection(name)
            except: pass
        mrag._init_collections()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
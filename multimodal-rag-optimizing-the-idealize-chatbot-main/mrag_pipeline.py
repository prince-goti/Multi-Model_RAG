import os
import hashlib
import nltk
import torch
import fitz 
import io
from typing import List, Dict
from PIL import Image

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from rank_bm25 import BM25Okapi

def download_nltk():
    resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if 'punkt' in res else f'taggers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)
download_nltk()

try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

class MRAGPipeline:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        print("\n⚡ Initializing MRAG (Deep Description Mode)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   ► Acceleration: {self.device.upper()}")

        self.client = QdrantClient(url=qdrant_url)
        self.collections = {
            'pdf': 'mrag_pdf',
            'image': 'mrag_image',
            'code': 'mrag_code'
        }
        
        print("   ► Loading BGE-M3 (Text/Code)...")
        self.dense_model = SentenceTransformer('BAAI/bge-m3', device=self.device)
        
        print("   ► Loading Vision Stack (CLIP + BLIP)...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

        self.bm25_indices = {'pdf': None, 'code': None}
        self.bm25_corpus = {'pdf': [], 'code': []}
        
        self._init_collections()
        print("✓ System Ready\n")

    def _init_collections(self):

        configs = {
            'mrag_pdf': 1024,   
            'mrag_image': 512,  
            'mrag_code': 1024  
        }
        
        for name, size in configs.items():
            try:
                self.client.get_collection(name)
            except:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=size, distance=Distance.COSINE)
                )

    def get_stats(self):
        stats = {}
        try:
            for key, col in self.collections.items():
                stats[key] = {"count": self.client.count(col).count}
        except:
            stats = {"error": "DB Disconnected"}
        return stats

    def _generate_deep_description(self, image):
        text = "a detailed description of the diagram showing"
        inputs = self.caption_processor(image, text=text, return_tensors="pt").to(self.device)
        
        out = self.caption_model.generate(
            **inputs, 
            max_new_tokens=150, 
            min_length=40,       
            num_beams=4,         
            early_stopping=True
        )
        return self.caption_processor.decode(out[0], skip_special_tokens=True)


    def ingest_pdf(self, path: str) -> Dict:
        print(f"Processing PDF: {os.path.basename(path)}")
        text_chunks_count = 0
        image_chunks_count = 0
        
        text = ""
        if UNSTRUCTURED_AVAILABLE:
            try:
                elements = partition_pdf(filename=path, strategy="fast")
                text = "\n\n".join([str(e) for e in elements])
            except: pass
        if not text:
            try:
                import pypdf
                reader = pypdf.PdfReader(path)
                text = "\n".join([p.extract_text() for p in reader.pages])
            except Exception as e: return {"error": str(e)}

        chunks = [text[i:i+600] for i in range(0, len(text), 600)]
        points = []
        for idx, chunk in enumerate(chunks):
            vec = self.dense_model.encode(chunk).tolist()
            pid = int(hashlib.md5(f"{path}_{idx}".encode()).hexdigest()[:16], 16) % (2**63)
            points.append(PointStruct(
                id=pid, vector=vec,
                payload={"text": chunk, "source": os.path.basename(path), "type": "pdf_text"}
            ))
            self.bm25_corpus['pdf'].append(chunk.split())

        if points:
            self.client.upsert(self.collections['pdf'], points)
            self.bm25_indices['pdf'] = BM25Okapi(self.bm25_corpus['pdf'])
            text_chunks_count = len(points)

        print("Extracting and Describing PDF Visuals...")
        try:
            image_chunks_count = self._extract_and_ingest_pdf_images(path)
        except Exception as e:
            print(f"Image extraction warning: {e}")

        return {"chunks": text_chunks_count, "images_extracted": image_chunks_count, "type": "pdf"}

    def _extract_and_ingest_pdf_images(self, path: str) -> int:
        doc = fitz.open(path)
        count = 0
        for page_num, page in enumerate(doc):
            for img in page.get_images(full=True):
                try:
                    xref = img[0]
                    base = doc.extract_image(xref)
                    image = Image.open(io.BytesIO(base["image"])).convert("RGB")
                    if image.width < 150 or image.height < 150: continue

                    caption = self._generate_deep_description(image)
                    
                    inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                    vec = self.clip_model.get_image_features(**inputs)[0].detach().cpu().numpy().tolist()
                    
                    pid = int(hashlib.md5(f"{path}_{page_num}_{xref}".encode()).hexdigest()[:16], 16) % (2**63)
                    self.client.upsert(
                        self.collections['image'],
                        points=[PointStruct(
                            id=pid, vector=vec,
                            payload={
                                "text": f"Visual from {os.path.basename(path)} (Page {page_num+1}): {caption}",
                                "source": os.path.basename(path), "type": "pdf_visual"
                            }
                        )]
                    )
                    count += 1
                    print(f"Visual (Pg {page_num+1}): {caption[:60]}...")
                except: continue
        return count

    def ingest_code(self, path: str) -> Dict:
        print(f"Processing Code: {os.path.basename(path)}")
        with open(path, 'r', encoding='utf-8', errors='ignore') as f: code = f.read()
        chunks = [code[i:i+800] for i in range(0, len(code), 800)]
        points = []
        for idx, chunk in enumerate(chunks):
            vec = self.dense_model.encode(chunk).tolist()
            pid = int(hashlib.md5(f"{path}_{idx}".encode()).hexdigest()[:16], 16) % (2**63)
            points.append(PointStruct(
                id=pid, vector=vec, 
                payload={"text": chunk, "source": os.path.basename(path), "type": "code"}
            ))
            self.bm25_corpus['code'].append(chunk.split())
        if points:
            self.client.upsert(self.collections['code'], points)
            self.bm25_indices['code'] = BM25Okapi(self.bm25_corpus['code'])
        return {"chunks": len(points), "type": "code"}

    def ingest_image(self, path: str) -> Dict:
        print(f"Deep Analyzing Image: {os.path.basename(path)}")
        image = Image.open(path).convert('RGB')
        

        caption = self._generate_deep_description(image)
        print(f"   ► Analysis: {caption}")
        
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        vec = self.clip_model.get_image_features(**inputs)[0].detach().cpu().numpy().tolist()
            
        pid = int(hashlib.md5(path.encode()).hexdigest()[:16], 16) % (2**63)
        self.client.upsert(
            self.collections['image'],
            points=[PointStruct(
                id=pid, vector=vec,
                payload={"text": f"[Visual Evidence from {os.path.basename(path)}]: {caption}", "source": os.path.basename(path), "type": "image"}
            )]
        )
        return {"chunks": 1, "type": "image", "caption": caption}


    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        results = []
        boost_code = 1.2 if any(x in query.lower() for x in ['code', 'python', 'function']) else 1.0
        boost_visual = 1.5 if any(x in query.lower() for x in ['image', 'diagram', 'chart', 'show']) else 1.0

        try:
            vec = self.dense_model.encode(query).tolist()
            hits = self.client.search(self.collections['pdf'], vec, limit=top_k)
            results.extend([{"text": h.payload.get('text', ''), "score": h.score, "source": h.payload.get('source'), "type": "pdf"} for h in hits])
        except Exception as e: print(f"⚠ PDF Search: {e}")

        try:
            hits = self.client.search(self.collections['code'], vec, limit=top_k)
            results.extend([{"text": h.payload.get('text', ''), "score": h.score * boost_code, "source": h.payload.get('source'), "type": "code"} for h in hits])
        except: pass

        try:
            inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True).to(self.device)
            vec_img = self.clip_model.get_text_features(**inputs)[0].detach().cpu().numpy().tolist()
            hits = self.client.search(self.collections['image'], vec_img, limit=4)
            results.extend([{"text": h.payload.get('text', ''), "score": h.score * boost_visual, "source": h.payload.get('source'), "type": "image"} for h in hits])
        except Exception as e: print(f"⚠ Visual Search: {e}")

        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k*2]
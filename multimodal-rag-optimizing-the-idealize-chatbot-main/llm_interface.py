import httpx
import json
import re
from typing import List, Dict, Any

class LLMInterface:
   
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "qwen2.5:7b"
    
        try:
            response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                print("Connected to Ollama")
            else:
                print("Ollama connection issue")
        except Exception as e:
            print(f"Could not connect to Ollama: {e}")
    
    def rerank_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
    
        print(f"\nRe-ranking {len(candidates)} candidates...")
        
        if not candidates:
            return []
        
        candidates_text = ""
        for idx, cand in enumerate(candidates[:50]): 
            source = cand.get('source', 'unknown')
            modality = cand.get('type', 'unknown') 
            text_preview = cand['text'][:250].replace('\n', ' ')
            
            candidates_text += f"\n[{idx}] Source: {source} | Type: {modality}\n{text_preview}\n"
        
        rerank_prompt = f"""You are a relevance scoring system.

Query: {query}

Candidates:
{candidates_text}

Task: Score each candidate [0-10] based on relevance to the Query.
Return ONLY a JSON array of scores, e.g., [9.5, 3.2, 7.0]. No explanation."""

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": rerank_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 500
                    }
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                scores_text = result['response'].strip()
                scores = self._extract_scores(scores_text, len(candidates[:50]))
                
                for idx, score in enumerate(scores):
                    if idx < len(candidates):
                        candidates[idx]['rerank_score'] = score
                
                ranked = sorted(
                    candidates[:50],
                    key=lambda x: x.get('rerank_score', 0),
                    reverse=True
                )
                
                print(f"Re-ranked {len(ranked)} candidates")
                return ranked
            
        except Exception as e:
            print(f"Re-ranking failed: {e}")
            return candidates[:10]
        
        return candidates[:10]
    
    def _extract_scores(self, text: str, expected_count: int) -> List[float]:
        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end != -1:
                return [float(s) for s in json.loads(text[start:end])]
        except:
            pass
        
        numbers = re.findall(r'\d+\.?\d*', text)
        scores = [float(n) for n in numbers[:expected_count]]
        while len(scores) < expected_count:
            scores.append(5.0)
        return scores[:expected_count]
    
    def generate_answer(self, query: str, context: List[Dict]) -> Dict[str, Any]:
        print(f"\nGenerating answer with {len(context)} context chunks...")
        
        context_text = ""
        sources = []
        
        for idx, ctx in enumerate(context[:7]): 
            text = ctx['text']
            source = ctx.get('source', 'unknown')
            c_type = ctx.get('type', 'unknown')
            
            if "Image Description:" in text or c_type == 'image':
                context_text += f"\n[Visual Evidence from {source}]: {text}\n"
            else:
                context_text += f"\n[Document Context from {source}]: {text}\n"
            
            sources.append(f"{source} ({c_type})")


        prompt = f"""
        You are an advanced Multimodal Research Assistant built with Late-Fusion Architecture.
        
        INSTRUCTIONS:
        1. Answer the USER QUESTION based ONLY on the provided CONTEXT.
        2. The CONTEXT contains text documents and VISUAL EVIDENCE (descriptions of images).
        3. IMPORTANT: If you see '[Visual Evidence]', treat that text as a direct description of an image the user provided. Do NOT say "I cannot see the image". Instead, describe the diagram/image based on that evidence.
        4. Cite your sources using.
        5. Maintain a professional, academic tone suitable for a university presentation.
        
        CONTEXT:
        {context_text}
        
        USER QUESTION: 
        {query}
        
        ANSWER:
        """

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_ctx": 4096
                    }
                },
                timeout=120.0
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['response'].strip()
                print(f"Generated answer ({len(answer)} chars)")
                
                return {
                    "answer": answer,
                    "sources": list(set(sources)),
                    "context_used": len(context),
                    "model": self.model,
                   
                    "evidence": [
                        {
                            "source": c.get('source', 'unknown'),
                            "score": c.get('rerank_score', 0),
                            "text": c.get('text', '')[:200] + "...",
                            "type": c.get('type', 'unknown')
                        }
                        for c in context[:5]
                    ]
                }
        
        except Exception as e:
            print(f"Generation failed: {e}")
            return { "answer": f"Error: {str(e)}", "sources": [], "context_used": 0, "model": self.model }
        
        return { "answer": "Failed to generate answer", "sources": [], "context_used": 0, "model": self.model }
    
    def query(self, query_text: str, candidates: List[Dict]) -> Dict[str, Any]:
        ranked_candidates = self.rerank_candidates(query_text, candidates)
        result = self.generate_answer(query_text, ranked_candidates)
        result['candidates_retrieved'] = len(candidates)
        result['candidates_reranked'] = len(ranked_candidates)
        return result
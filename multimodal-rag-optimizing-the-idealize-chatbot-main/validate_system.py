import requests
import json
import time
from datetime import datetime

TEST_CASES = [
    {
        "category": "Architecture",
        "query": "What is the F1 score of Late-Fusion compared to Early-Fusion?",
        "expected_keyword": "0.867"
    },
    {
        "category": "Code Retrieval",
        "query": "How does GTE-Qwen improve code retrieval performance?",
        "expected_keyword": "hybrid"
    },
    {
        "category": "Vision",
        "query": "What are the limitations of CLIP models mentioned in the text?",
        "expected_keyword": "bias"
    },
    {
        "category": "Infrastructure",
        "query": "Why is Qdrant selected as the vector database?",
        "expected_keyword": "hybrid"
    }
]

def run_validation():
    print("Starting MRAG Validation Protocol...")
    results = []
    
    for test in TEST_CASES:
        print(f"\nTesting Category: {test['category']}")
        print(f"Query: {test['query']}")
        
        start_time = time.time()
        try:
            response = requests.post(
                "http://localhost:5000/query",
                json={"message": test['query']},
                timeout=120
            )
            data = response.json()
            duration = time.time() - start_time
            
            answer = data.get('response', '')
            sources = data.get('sources', [])
            
            passed = test['expected_keyword'].lower() in answer.lower()
            
            result_entry = {
                "category": test['category'],
                "query": test['query'],
                "response_preview": answer[:100] + "...",
                "sources_count": len(sources),
                "latency_seconds": round(duration, 2),
                "status": "PASS" if passed else "REVIEW"
            }
            results.append(result_entry)
            print(f"Status: {result_entry['status']} ({duration:.2f}s)")
            
        except Exception as e:
            print(f"Error: {e}")
            
    with open("validation_report.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nValidation Complete. Report saved to 'validation_report.json'")

if __name__ == "__main__":
    run_validation()
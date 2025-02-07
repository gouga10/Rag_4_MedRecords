from rag import get_index, generate_answer
from pydantic import BaseModel
import time
from typing import Dict,Any
from fastapi import FastAPI
from pydantic import BaseModel
import openai

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    conv: str = None


@app.post("/generate")
async def generate_answer_endpoint(request: QueryRequest) -> Dict[str, Any]:
    """Generate an answer for a given book and question."""
    answer = generate_answer(
            index=index,
            question=request.query,
            prev_conversation=request.conv,
            topk=4
        )        
    
    return {"response":str(answer)}


@app.get("/api/health", tags=["Health"])
async def health_check():
    """Check API health and loaded books."""
    return {"status": "ok","index" :'loaded' }
    


if __name__ == "__main__":
    import uvicorn
    import os
    import dotenv
    
    dotenv.load_dotenv()
    try:
        os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
        index=get_index()
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except Exception as e:
        print(e)
    
    
    
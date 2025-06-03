from fastapi import APIRouter, UploadFile, File
from src.utils.evaluation.ragas_evaluator import evaluate_with_ragas
from src.utils.evaluation.evaluator import Evaluator
import json

router = APIRouter()

@router.post("/ragas")
async def ragas_eval(file: UploadFile = File(...)):
    contents = await file.read()
    data = json.loads(contents)
    results = evaluate_with_ragas(data)
    return results

@router.post("/classical")
async def classical_eval(file: UploadFile = File(...)):
    contents = await file.read()
    data = json.loads(contents)
    results = Evaluator(data)
    return results

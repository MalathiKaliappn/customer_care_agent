import os
import asyncio
import logging
from typing import Dict, Any

import openai
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_recall,
    context_precision,
    faithfulness,
)
import backoff
import requests
from httpx import HTTPStatusError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
RAGAS_MODEL = os.environ.get("RAGAS_MODEL", "gpt-3.5-turbo")  # Default fallback

# Set OpenAI API key globally
openai.api_key = OPENAI_API_KEY

# Async semaphore for concurrency control
MAX_CONCURRENCY = 3
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


@backoff.on_exception(backoff.expo, HTTPStatusError, max_tries=5)
def mistral_eval_call(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Sync function to call Mistral API with retry logic."""
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
    }
    payload = {
        "model": RAGAS_MODEL,
        "messages": [
            {"role": "system", "content": inputs["system"]},
            {"role": "user", "content": inputs["question"]},
        ],
        "temperature": 0.0,
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 429:
        logger.warning("429 Too Many Requests")
        response.raise_for_status()

    data = response.json()
    return {"answer": data["choices"][0]["message"]["content"]}


async def mistral_async_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Async wrapper for mistral_eval_call with concurrency limiting."""
    async with semaphore:
        return await asyncio.to_thread(mistral_eval_call, inputs)


def get_eval_model(model_name: str):
    """Return the evaluation model callable based on the model name."""
    if "gpt" in model_name:
        return ChatOpenAI(model=model_name, temperature=0.0)
    else:
        return mistral_async_fn


async def evaluate_with_ragas(dataset):
    """Run evaluation on given dataset with RAGAS metrics and return result."""
    eval_model = get_eval_model(RAGAS_MODEL)
    logger.info(f"Evaluating with model: {RAGAS_MODEL}")

    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=eval_model,
    )

    logger.info("Evaluation Complete.")
    return result


async def main():
    from ragas_dataset import dataset  # Adjust import as needed

    result = await evaluate_with_ragas(dataset)
    print(result.to_pandas())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user.")

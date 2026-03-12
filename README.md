# LLM-Straive

A Python project providing a production-ready FastAPI service for text summarization using LLMs GROQ_API_KEY.

## Project Purpose

This project implements:

- **Production API**: A FastAPI service with a POST /summarize endpoint. It accepts a string and a "style" preference, calls an LLM (OpenAI/Anthropic) for summarization, and logs all failures with timestamps. The solution is non-blocking, avoids hardcoded API keys, and is designed for production use.
- **Batch Challenge**: An endpoint POST /batch-process that processes a list of 1,000 text paragraphs efficiently through the LLM, handling large volumes and API limits. Returns a JSON summary of success vs. failure counts.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/diya0426/straive-task1.git
   ```
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```
uvicorn main:app --reload
```

## Requirements

See requirements.txt for dependencies.

## Author

indhumathi.K

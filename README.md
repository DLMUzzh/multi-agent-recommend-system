# Multi-Agent Recommendation and QA System

一个基于 FastAPI、LangGraph、SQLite 和结构化 LLM 调用的多智能体推荐与知识问答服务。

## Features

- Conversation-based document recommendation
- Knowledge retrieval and evidence-grounded answers
- Similar-document recommendation API
- User interaction memory and feedback recovery
- SQLite-backed local persistence
- Offline regression and evaluation tests

## Project Layout

```text
python/app/       Application source code
python/assess/    Regression and evaluation tests
Test/             Additional test utilities
data/             Public synthetic evaluation fixtures
scripts/          Local verification scripts
```

## Run Locally

Install dependencies, provide runtime credentials through local environment variables, and start the service:

```bash
python -m pip install -r python/requirements.txt
python -m uvicorn app.main:app --app-dir python --host 0.0.0.0 --port 8000
```

The service reads credentials from the local environment. Do not commit `.env` files, API keys, passwords, tokens, private keys, databases, or runtime data.

## Verification

```bash
bash scripts/verify.sh
```

The verification suite is designed to run without live external LLM calls.

## License

Add the project license before distributing this repository publicly.

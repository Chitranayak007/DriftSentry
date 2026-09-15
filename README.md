# DriftSentry AI — API-Key-Free Integrated Prototype

This version includes the behavioral ML engine, NetworkX graph analysis, decay-weighted risk scoring, ContextGuard, functional frontend views, and a deterministic SOC Storyline Generator.

## Important
Gemini has been completely removed from the backend. No Gemini API key, `.env`, or external LLM service is required. The SOC storyline is generated from the actual ML signals, graph changes, risk level, and ContextGuard result.

## Run backend
```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## Run frontend
In another terminal:
```bash
cd frontend
python -m http.server 5500
```

Open: http://127.0.0.1:5500

## GitHub
There is no secret API key to commit. The project can be pushed to GitHub normally.

## Demo architecture
- **ML engine:** Isolation Forest + interpretable behavioral signals
- **Graph engine:** NetworkX-based temporal access/relationship drift
- **Risk engine:** combines ML, graph drift, context correction, and foreign-location signal
- **ContextGuard:** checks mock HR/Jira/GitHub justification
- **SOC Storyline:** deterministic, explainable Python-generated narrative

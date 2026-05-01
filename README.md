# Autonomous Finance Tutor

Production-structured, beginner-friendly project using:

- Gemini API (Google AI Studio)
- CrewAI (Planner + Tutor agents)
- LangChain + ChromaDB (RAG)
- Streamlit frontend
- Docker
- GitHub Actions CI
- Kubernetes deployment

## Project Structure

src/

- agents/
  - planner.py
  - tutor.py
- tools/
  - retriever.py
- utils/
  - logger.py
- app.py

## Local Run

1. Install dependencies:

	pip install -r requirements.txt

2. Start Streamlit app:

	streamlit run src/app.py

3. Open app in browser (usually http://localhost:8501).

4. In sidebar:
	- Add Gemini API key
	- Upload finance PDF
	- Click Ingest PDF

5. Ask a finance question in chat.

## Docker Run

1. Build image:

	docker build -t fin-agent .

2. Run container:

	docker run -p 8501:8501 fin-agent

## Kubernetes Run (Minikube or K3d)

1. Build image locally for your cluster runtime.
2. Apply manifest:

	kubectl apply -f k8s/deployment.yaml

3. Check resources:

	kubectl get pods
	kubectl get svc

## CI

GitHub Actions workflow at .github/workflows/ci.yml runs:

- Ruff lint on src
- Docker build verification

## Error Handling Implemented

- Missing API key -> Streamlit warning
- No PDF uploaded -> Streamlit alert
- Empty retrieval -> fallback warning and general tutor response
- API failures -> one retry with error shown on failure

"""Entry point to launch the gem5 ML Prediction Web Dashboard & API."""
import os
import sys
import uvicorn

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="gem5 ML Performance Prediction UI & Backend Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"🚀 gem5 ML Performance Predictor Web UI & REST API")
    print(f"📡 Dashboard available at: http://{args.host}:{args.port}")
    print(f"📚 OpenAPI interactive docs: http://{args.host}:{args.port}/docs")
    print(f"=======================================================\n")

    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=args.reload)

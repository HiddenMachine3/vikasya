import argparse

from backend.api import ApiFrontend
from backend.media_analyzer import MediaAnalyzerService
from frontend.cli_frontend import CliFrontend
from frontend.streamlit_frontend import StreamlitFrontend


def main():
    """Initializes backend and frontend, then runs the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vikasya Deepfake Detection System")
    parser.add_argument(
        "--frontend",
        choices=["cli", "streamlit", "api"],
        default="cli",
        help="Frontend to use (cli, streamlit, or api)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address for API server (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port", type=int, default=8000, help="Port for API server (default: 8000)"
    )

    args = parser.parse_args()

    # Initialize the backend service
    analyzer_service = MediaAnalyzerService()

    # Choose and initialize the frontend based on arguments
    if args.frontend == "cli":
        frontend = CliFrontend(analyzer_service)
    elif args.frontend == "streamlit":
        frontend = StreamlitFrontend(analyzer_service)
    elif args.frontend == "api":
        frontend = ApiFrontend(analyzer_service)
        return frontend.run(host=args.host, port=args.port)

    # Run the application
    frontend.run()


if __name__ == "__main__":
    main()

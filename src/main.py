from backend.media_analyzer import MediaAnalyzerService
from frontend.streamlit_frontend import StreamlitFrontend
# from frontend.cli_frontend import CliFrontend

def main():
    """Initializes backend and frontend, then runs the application."""
    # Initialize the backend service
    analyzer_service = MediaAnalyzerService()

    # Choose and initialize the frontend
    # Currently using CLI, can be swapped later
    # frontend = CliFrontend(analyzer_service)
    frontend = StreamlitFrontend(analyzer_service)  # Example of swapping
    # frontend = GuiFrontend(analyzer_service) # Example of swapping
    # Run the application
    frontend.run()

if __name__ == "__main__":
    main()

from .base_frontend import BaseFrontend
from backend.media_analyzer import MediaAnalyzerService
import pprint

class CliFrontend(BaseFrontend):
    """Command Line Interface frontend implementation."""

    def __init__(self, analyzer_service: MediaAnalyzerService):
        self.analyzer_service = analyzer_service
        self.pretty_printer = pprint.PrettyPrinter(indent=2)

    def get_file_path(self) -> str | None:
        """Gets file path from user input."""
        try:
            path = input("Enter the path to the media file (or type 'quit' to exit): ")
            if path.lower() == 'quit':
                return None
            return path
        except EOFError: # Handle Ctrl+D
            return None


    def display_results(self, results: list[dict]):
        """Prints analysis results to the console."""
        print("\\n--- Analysis Results ---")
        if not results:
            print("No analysis was performed.")
            return

        for result in results:
             if "error" in result:
                 print(f"ERROR: {result['error']}")
             else:
                 print(f"Analyzer: {result.get('analyzer', 'Unknown')}")
                 self.pretty_printer.pprint(result)
                 print("-" * 20)
        print("--- End of Results ---\\n")


    def run(self):
        """Runs the CLI interaction loop."""
        print("--- Media Authenticity Verification CLI ---")
        while True:
            file_path = self.get_file_path()
            if file_path is None:
                print("Exiting.")
                break

            if not file_path:
                print("Please enter a valid file path.")
                continue

            analysis_results = self.analyzer_service.analyze_media(file_path)
            self.display_results(analysis_results)


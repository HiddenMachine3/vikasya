import os
import tempfile

import streamlit as st

from backend.media_analyzer import MediaAnalyzerService

from .base_frontend import BaseFrontend


class StreamlitFrontend(BaseFrontend):
    """Streamlit frontend implementation."""

    def __init__(self, analyzer_service: MediaAnalyzerService):
        self.analyzer_service = analyzer_service

    def get_file_path(self) -> str | None:
        """Gets file path from Streamlit file uploader."""
        uploaded_file = st.file_uploader(
            "Choose a media file (image or video)",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "wav", "flac", "mp3"],
        )
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location to get a file path
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                return tmp_file.name
        return None

    def display_results(self, results: list[dict]):
        """Displays analysis results using Streamlit components."""
        st.subheader("Analysis Results")
        if not results:
            st.info("No analysis was performed or returned results.")
            return

        # Display predictions prominently first if they exist
        for result in results:
            if "prediction" in result:
                st.header(f"Prediction: {result['prediction']}")

        # Display detailed results in expanders
        for result in results:
            if "error" in result:
                st.error(
                    f"Analyzer: {result.get('analyzer', 'Unknown')} - ERROR: {result['error']}"
                )
            else:
                # Avoid displaying prediction again in the expander if already shown above
                display_data = {k: v for k, v in result.items() if k != "prediction"}
                if display_data:  # Only show expander if there's other data
                    with st.expander(
                        f"Details from: {result.get('analyzer', 'Unknown')}",
                        expanded=True,
                    ):
                        st.json(display_data)  # Display the filtered dictionary as JSON

    def run(self):
        """Runs the Streamlit application interface."""
        st.title("Media Authenticity Verification")

        temp_file_path = self.get_file_path()

        if temp_file_path:
            # Display the uploaded media
            file_type = os.path.splitext(temp_file_path)[1].lower()
            if file_type in [".jpg", ".jpeg", ".png"]:
                st.image(temp_file_path, caption="Uploaded Image")
            elif file_type in [".mp4", ".avi", ".mov"]:
                st.video(temp_file_path)
            elif file_type in [".wav", ".flac", ".mp3"]:
                st.audio(temp_file_path)
            else:
                st.warning("Unsupported file type for preview.")

            if st.button("Analyze Media"):
                with st.spinner("Analyzing..."):
                    try:
                        analysis_results = self.analyzer_service.analyze_media(
                            temp_file_path
                        )
                        self.display_results(analysis_results)
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
        else:
            st.info("Please upload a media file to begin analysis.")


# Note: To run this, you would typically have a main script like:
# if __name__ == "__main__":
#     # Configure your analyzers here
#     analyzer_config = {"metadata": True, "ela": False, ...}
#     service = MediaAnalyzerService(analyzer_config)
#     frontend = StreamlitFrontend(service)
#     frontend.run()
# And run it using `streamlit run your_main_script.py`

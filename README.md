# Media Authenticity Verification App (CLI Version)

This project provides a command-line tool to analyze media files (images and videos) for potential signs of manipulation using various techniques like Fractal Density Analysis, Error Level Analysis (ELA), Chromatic Aberration detection, and basic video inconsistency checks.

## Prerequisites

*   Python 3.7+
*   `pip` (Python package installer)

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Activate the virtual environment** (if you created one):
    ```bash
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```
2.  **Navigate to the project's root directory** (the directory containing `src/` and `requirements.txt`).
3.  **Run the main script:**
    ```bash
    python src/main.py
    ```

## How to Use

1.  When you run the application using `python src/main.py`, you will see the welcome message: `--- Media Authenticity Verification CLI ---`
2.  The application will then prompt you: `Enter the path to the media file (or type 'quit' to exit):`
3.  Provide the complete path to the image or video file you want to analyze and press Enter.
    *   Supported image types include `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`.
    *   Supported video types include `.mp4`, `.avi`, `.mov`, `.mkv`.
4.  The tool will process the file using the available analyzers and display the results for each one.
5.  To stop the application, type `quit` at the prompt and press Enter.

## Current Features

*   **Image Analysis:**
    *   Fractal Density Analysis (Placeholder)
    *   Error Level Analysis (ELA) (Basic Implementation)
    *   Chromatic Aberration Detection (Placeholder)
*   **Video Analysis:**
    *   Basic Frame/Duration Info & Heuristics (Placeholder)
*   **CLI Interface:** Simple command-line interaction.
*   **Decoupled Design:** Backend analysis is separated from the frontend interface.

## Future Work

*   Implement more robust analysis algorithms.
*   Add audio analysis capabilities.
*   Develop a graphical user interface (GUI) or mobile app frontend.
*   Integrate deep learning models for detection.

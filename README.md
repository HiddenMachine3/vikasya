# Media Authenticity Verification App

This project provides tools to analyze media files (images, videos, and audio) for potential signs of manipulation using various techniques. It includes a command-line interface (CLI) and a Flutter-based mobile application.

## Prerequisites

*   Python 3.7+ (for CLI version)
*   `pip` (Python package installer)
*   Flutter SDK (for the mobile app)

## Setup

### CLI Version

1.  **Clone the repository:**
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

### Flutter App

1.  **Navigate to the `deepfake` folder:**
    ```bash
    cd deepfake
    ```

2.  **Install Flutter dependencies:**
    ```bash
    flutter pub get
    ```

3.  **Run the app:**
    ```bash
    flutter run
    ```

## Running the Application

### CLI Version

1.  **Activate the virtual environment** (if you created one):
    ```bash
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **Navigate to the project's root directory** (the directory containing `src/` and `requirements.txt`).

3.  **Run the main script:**
    ```bash
    python src/main.py
    ```

### Flutter App

1.  **Ensure Flutter is installed and configured.**

2.  **Navigate to the `deepfake` folder and run the app:**
    ```bash
    cd deepfake
    flutter run
    ```

## How to Use

### CLI Version

1.  When you run the application using `python src/main.py`, you will see the welcome message: `--- Media Authenticity Verification CLI ---`
2.  The application will then prompt you: `Enter the path to the media file (or type 'quit' to exit):`
3.  Provide the complete path to the image, video, or audio file you want to analyze and press Enter.
    *   Supported image types include `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`.
    *   Supported video types include `.mp4`, `.avi`, `.mov`, `.mkv`.
    *   Supported audio types include `.wav`, `.mp3`.
4.  The tool will process the file using the available analyzers and display the results for each one.
5.  To stop the application, type `quit` at the prompt and press Enter.

### Flutter App

1.  Launch the app on your device or emulator.
2.  Use the app's interface to upload and analyze media files for deepfake detection.

## Current Features

*   **Image Analysis:**
    *   Fractal Density Analysis (Placeholder)
    *   Error Level Analysis (ELA) (Basic Implementation)
    *   Chromatic Aberration Detection (Placeholder)
*   **Video Analysis:**
    *   Basic Frame/Duration Info & Heuristics (Placeholder)
*   **Audio Analysis:**
    *   Basic audio spectrum analysis (using FFT)
    *   WAV file processing
*   **Flutter App:**
    *   Mobile app for deepfake detection (image, video, and audio)
    *   Built using Flutter with support for Android, iOS, macOS, and web.
*   **CLI Interface:** Simple command-line interaction.
*   **Decoupled Design:** Backend analysis is separated from the frontend interface.

## Future Work

*   Implement more robust analysis algorithms.
*   Enhance audio analysis capabilities.
*   Improve the Flutter app's UI/UX.
*   Integrate deep learning models for detection.
*   Add support for additional media formats.

## Additional Notes

The Flutter app is located in the `deepfake` folder and includes a TFLite model for deepfake detection. Refer to the `deepfake/README.md` for more details on the Flutter project.

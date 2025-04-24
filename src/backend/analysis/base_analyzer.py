\
from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    """Abstract base class for all media analysis techniques."""

    @abstractmethod
    def analyze(self, file_path: str) -> dict:
        """
        Analyzes the given media file.

        Args:
            file_path: The path to the media file (image or video).

        Returns:
            A dictionary containing the analysis results.
        """
        pass

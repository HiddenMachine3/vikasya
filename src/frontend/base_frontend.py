\
from abc import ABC, abstractmethod

class BaseFrontend(ABC):
    """Abstract base class for different frontend implementations."""

    @abstractmethod
    def run(self):
        """Starts the frontend interaction loop."""
        pass

    @abstractmethod
    def display_results(self, results: list[dict]):
        """
        Displays the analysis results to the user.

        Args:
            results: A list of dictionaries containing analysis results.
        """
        pass

    @abstractmethod
    def get_file_path(self) -> str | None:
        """
        Prompts the user for a media file path.

        Returns:
            The file path entered by the user, or None if they wish to exit.
        """
        pass


"""Test configuration for pytest."""

# Suppress warnings during tests
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", ".*dataloader.*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")

from pathlib import Path

from dotenv import load_dotenv
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "local dev"

    ...
dotenv_local_path = Path(__file__).parent / ".env.local"
dotenv_path_example = Path(__file__).parent / ".env"

if dotenv_local_path.is_file():
    load_dotenv(dotenv_path=dotenv_local_path)
    print("Configuration Loaded from .env.local")

else:
    load_dotenv(dotenv_path=dotenv_path_example)
    print("Configuration Loaded from .env")

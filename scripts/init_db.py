from app.state_manager import StateManager
from app.utils import ensure_directories


def main() -> None:
    try:
        ensure_directories()
        StateManager()
        print("Initialized data directories, registry, and SQLite schema.")
    except Exception as e:
        print("Initialization failed:")
        print(str(e))
        raise


if __name__ == "__main__":
    main()
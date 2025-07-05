import sys
from tracker import Tracker
from config import load_config

def main():
    try:
        config = load_config("settings.ini")
        tracker = Tracker(**config)
        tracker.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
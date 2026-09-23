"""Run one background research worker: python worker.py"""
from dotenv import load_dotenv


def main():
    load_dotenv()
    from core.background import BackgroundWorker
    BackgroundWorker().run_forever()


if __name__ == "__main__":
    main()

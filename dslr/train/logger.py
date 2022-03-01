import colorama
import logging.handlers
from colorama import Fore, Style


class Logger:

    Logger = None
    colorama.init()

    def __init__(self, level, name="truc"):
        self.Logger = logging.getLogger(name)
        self.Logger.setLevel(level)
        formatter = logging.Formatter("%(levelname)s - %(message)s")

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.Logger.addHandler(ch)

    def info(self, message):
        self.Logger.info(message)

    def warning(self, message):
        self.Logger.warning(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")

    def error(self, message):
        self.Logger.error(f"{Fore.RED}{message}{Style.RESET_ALL}")
        exit()

    def debug(self, message):
        self.Logger.debug(message)

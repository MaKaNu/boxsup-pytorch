"""Config Module."""

from ast import literal_eval
import configparser
from pathlib import Path


class GLOBAL_CONFIG:
    """Config Class for global config providing."""

    __conf = None
    __path = Path(__file__).resolve().parent / "example.ini"

    @staticmethod
    def config():
        """Load or return the saved config.

        Returns:
            _type_: the configuration
        """
        if GLOBAL_CONFIG.__conf is None:  # Read only once, lazy.
            GLOBAL_CONFIG.__conf = configparser.ConfigParser(
                converters={"any": lambda x: literal_eval(x)}
            )
            GLOBAL_CONFIG.__conf.read(GLOBAL_CONFIG.__path)
        return GLOBAL_CONFIG.__conf

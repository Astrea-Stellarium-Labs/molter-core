import typing

from .utils import escape_mentions


__all__ = ("CommandException", "BadArgument")


class CommandException(Exception):
    # if you already have this exception or similar, use that instead
    pass


class BadArgument(CommandException):
    """A special exception for invalid arguments when using prefixed commands."""

    def __init__(self, message: typing.Optional[str] = None, *args: typing.Any) -> None:
        if message is not None:
            message = escape_mentions(message)
            super().__init__(message, *args)
        else:
            super().__init__(*args)

"""Values representing common models in Discord Python libraries. These should be replaced by your own."""
import typing

__all__ = ("Client", "Message", "User", "Member", "Channel", "Guild", "MISSING")

Client: typing.Any
"""The client/bot model."""
Message: typing.Any
"""The Discord message model."""
User: typing.Any
"""The Discord user model."""
Member: typing.Any
"""The Discord member model."""
Channel: typing.Any
"""The Discord channel model."""
Guild: typing.Any
"""The Discord guild model."""


class MISSING:
    """
    A special class representing a non-value.

    This should be replaced by your own MISSING class or similar.
    """

    pass

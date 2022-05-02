import typing

import attrs

from .dummy import *

if typing.TYPE_CHECKING:
    from .command import PrefixedCommand

__all__ = ("PrefixedContext",)


@attrs.define(slots=True)
class PrefixedContext:
    """
    A special 'Context' object for prefixed commands.
    This should be adjusted and added onto for your needs.
    """

    client: "Client" = attrs.field()
    """The bot instance."""
    message: "Message" = attrs.field()
    """The message this represents."""
    author: "typing.Union[Member, User]" = attrs.field()
    """The person who sent the message."""

    channel: typing.Optional["Channel"] = attrs.field()
    """The channel this message was sent through."""
    guild: typing.Optional["Guild"] = attrs.field()
    """The guild this message was sent through."""

    invoked_name: str = attrs.field(default=None)
    """The name/alias used to invoke the command."""
    content_parameters: str = attrs.field(default=None)
    """The message content without the prefix or command."""
    command: "PrefixedCommand" = attrs.field(default=None)
    """The command invoked."""
    args: typing.List[str] = attrs.field(factory=list)
    """The arguments of the command (as a list of strings)."""
    prefix: str = attrs.field(default=None)
    """The prefix used for this command."""

    async def send(self, **kwargs):
        ...

    async def reply(self, **kwargs):
        ...

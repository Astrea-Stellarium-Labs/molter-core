import typing

from . import errors
from .context import PrefixedContext

T = typing.TypeVar("T")
T_co = typing.TypeVar("T_co", covariant=True)


@typing.runtime_checkable
class Converter(typing.Protocol[T_co]):
    """A protocol representing a class used to convert an argument."""

    async def convert(self, ctx: PrefixedContext, argument: str) -> T_co:
        """
        The function that converts an argument to the appropriate type.
        This should be overridden by subclasses for their conversion logic.

        Args:
            ctx: The context to use for the conversion.
            argument: The argument to be converted.

        Returns:
            Any: The converted argument.
        """
        raise NotImplementedError("Derived classes need to implement this.")


class _LiteralConverter(Converter):
    values: typing.Dict

    def __init__(self, args: typing.Any):
        self.values = {arg: type(arg) for arg in args}

    async def convert(self, ctx: PrefixedContext, argument: str):
        for arg, converter in self.values.items():
            try:
                if (converted := converter(argument)) == arg:
                    return converted
            except Exception:
                continue

        literals_list = [str(a) for a in self.values.keys()]
        literals_str = ", ".join(literals_list[:-1]) + f", or {literals_list[-1]}"
        raise errors.BadArgument(
            f'Could not convert "{argument}" into one of {literals_str}.'
        )


class Greedy(typing.List[T]):
    """A special marker class to mark an argument in a prefixed command to repeatedly convert until it fails to convert an argument."""


DISCORD_MODEL_TO_CONVERTER = {}
"""
A dictionary mapping of your library's Discord objects to their corresponding converters.

You will need to make these converters yourself.
"""

"""Exceptions for the icu-sepsis package."""


class InadmissibleActionError(Exception):
    """Raised when an inadmissible action is taken, and the strategy is to
    throw an exception."""

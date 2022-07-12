import warnings


# Section 1. Warnings
class AutoCorrectionWarning(Warning):
    """Warns about cases when the code is choosing new values automatically. Modified from GromacsWrapper."""


# Section 2. Errors
class ParameterError(Exception):
    """Error raised when detecting improperly specified parameters in the YAML file."""


class ParseError(Exception):
    """Error raised when parsing of a file failed. Modified from GromacsWrapper."""


# The warning should always be displayed because other parameters
# can have changed, eg during interactive use.
for w in (AutoCorrectionWarning,):
    warnings.simplefilter('always', category=w)
del w

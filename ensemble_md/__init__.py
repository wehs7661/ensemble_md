"""A repository for developing ensemble simulation methods"""

# Add imports here
# from .ensemble_md import canvas  # noqa: ABS101

# Handle versioneer
from ._version import get_versions  # noqa: ABS101

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

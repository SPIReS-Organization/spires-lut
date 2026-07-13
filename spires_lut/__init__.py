"""spires-lut: reflectance lookup tables for the SPIReS package family."""

__version__ = "0.1.0"

try:
    from . import disort
except ImportError:
    pass
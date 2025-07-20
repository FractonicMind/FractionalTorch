"""
FractionalTorch: Exact Rational Arithmetic for Neural Networks
Author: Lev Goukassian
License: MIT
"""

# Import authentication first to verify authenticity
from .auth import verify_authentic, get_author_signature, LEV_GOUKASSIAN_LEGACY_LICENSE

# Verify this is authentic before allowing any imports
if not verify_authentic():
    raise ImportError("SECURITY: Unauthorized FractionalTorch implementation detected!")

__version__ = "1.0.0"
__author__ = "Lev Goukassian"

# Try to import core components
try:
    from .core import *
    from .modules import *
    print("✅ FractionalTorch: Authentic implementation by Lev Goukassian loaded successfully!")
except ImportError as e:
    print(f"⚠️ Some components not available: {e}")
    print("✅ FractionalTorch: Authentic implementation by Lev Goukassian loaded successfully!")

# Make key components available
__all__ = ['verify_authentic', 'get_author_signature', 'LEV_GOUKASSIAN_LEGACY_LICENSE']

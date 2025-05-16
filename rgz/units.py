"""Stop-gap for supporting astropy units in static type-checking.

Units are generated at runtime, so static analysis tools can't find the units
in astropy.units.__init__. This is an open issue in astropy. One way around
this without fiddling with configuration is to enumerate all the units we
use and then ignore type errors in this file.

We only need to enumerate those units which we want to use in type signatures.

We can remove this code when the relevant issue is resolved:
https://github.com/astropy/astropy/issues/15808
"""

import astropy.units

# Import all the units so we can use this module as a drop-in substitute.
from astropy.units import *  # type: ignore[reportWildcardImportFromLibrary]

deg = astropy.units.deg  # type: ignore
arcmin = astropy.units.arcmin  # type: ignore
arcsec = astropy.units.arcsec  # type: ignore

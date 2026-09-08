"""Check periodic circular distance against a naive absolute difference.

Method:
    Complex-phase wrapping on a 2-pi-periodic circle.
Purpose:
    Print one wrap-around sanity comparison; this is not an automated test.
Inputs:
    Two fixed angles near opposite representations of the periodic seam.
Outputs:
    Two console values; no files are written.
Output location:
    None.
Dependencies:
    NumPy; no local module imports.
Related files:
    The stacking-potential generators contain related circular-distance helpers.
"""

import numpy as np


def circular_distance(a: float, b: float) -> float:
    """
    Return the shortest absolute angular distance |a-b| on a 2*pi-periodic circle.

    Angles are in radians. The result is always in [0, pi], so wrap-around
    cases 
    """
    return float(np.abs(np.angle(np.exp(1j * (a - b)))))


print(circular_distance(-np.pi + 0.1, np.pi - 0.1))
#compare with the naive distance that does not handle wrap-around correctly
print(np.abs((-np.pi + 0.1) - (np.pi - 0.1)))

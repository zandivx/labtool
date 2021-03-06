"""
A python library for scientific protocols at Technical University of Graz
GitHub: https://github.com/zandivx/labtool
"""

# dunders
__author__ = "Andreas Zach"
__version__ = "0.2.2"

try:
    # 3rd party library imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import uncertainties as u
    import uncertainties.unumpy as unp

except ImportError:
    raise ImportError("Requirements not satisfied!")

else:
    # own library imports
    from .src.classes import *
    from .src.functions import *
    from .src import monkeypatch_uncertainties

    # define __all__
    from .src.classes import __all__ as cls_all
    from .src.functions import __all__ as func_all
    __all__ = sorted(cls_all +
                     func_all +
                     ["np", "pd", "plt", "u", "unp"])  # type: ignore
    del cls_all, func_all

    # apply monkey patches
    monkeypatch_uncertainties.display()
    # monkeypatch_uncertainties.init()

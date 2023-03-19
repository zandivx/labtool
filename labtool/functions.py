"""labtool/src/functions.py"""

# dunders
__author__ = "Andreas Zach"
__all__ = [
    "cd",
    "plt_latex",
    "pd_format",
    "profile",
    "tracer",
    "plt_uplot",
    "separate_uarray",
    "integrate",
]

# std library
from cProfile import Profile
from os import chdir, path
from pstats import Stats, SortKey
from typing import Callable, Union, Iterable

# 3rd party
from matplotlib import rcParams
from matplotlib.pyplot import errorbar, fill_between, plot
from numpy import array, ndarray
from pandas import options
from scipy.integrate import trapezoid


def cd() -> None:
    """Change the current working directory to the directory of the calling script."""
    chdir(path.dirname(path.abspath(__file__)))
    return None


def plt_latex() -> None:
    """Use LaTeX as text processor for matplotlib, set figsize for a textwidth of 15 cm"""
    cm = 1 / 2.54  # conversion factor inch to cm
    rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{lmodern}\usepackage[locale=DE,uncertainty-mode=separate]{siunitx}",
            "font.family": "Latin Modern Roman",
            "figure.figsize": (15 * cm, 9 * cm),  # 15:9 relation
            "figure.autolayout": True,  # auto tight_layout()
        }
    )
    return None


def pd_format(format_spec: str) -> None:
    """Update float-formatting of pandas.DataFrame."""
    options.display.float_format = f"{{:{format_spec}}}".format  # type: ignore
    return None


def profile(func: Callable) -> Callable:
    """A decorator for profiling a certain function call"""

    def decorator(*args, **kwargs) -> None:
        with Profile() as pr:
            func(*args, **kwargs)
        stats = Stats(pr)
        stats.sort_stats(SortKey.TIME)
        stats.dump_stats(f"_profiling_{func.__name__}.snakeviz")
        stats.print_stats()

    return decorator


def tracer(frame, event, arg) -> None:
    """Copy from StackOverflow"""
    indent = [0]

    def list_arguments() -> None:
        try:
            for i in range(frame.f_code.co_argcount):
                name = frame.f_code.co_varnames[i]
                print(f"\tArgument {name} = {frame.f_locals[name]}")
        except Exception as e:
            string = f"EXCEPTION: {e}"
            line = "\n" + "-" * len(string) + "\n"
            print(line + string + line)

    if event == "call":
        indent[0] += 2
        print("-" * indent[0] + "> call function", frame.f_code.co_name)
        list_arguments()
    elif event == "return":
        print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
        indent[0] -= 2
        list_arguments()
    else:
        pass


def plt_uplot(
    x: Iterable,
    y: Iterable,
    *args,
    band: bool = True,
    kwfill: Union[dict, None] = None,
    **kwargs,
):
    """Take two uncertainties.unumpy.uarrays as input and plot them with matplotlib.pyplot.plot.
    Return the return value of either pyplot.fill_between or pyplot.errorbar

    Parameters:
    -> x\t\tuarray
    -> y\t\tuarray
    -> band\twheter to plot an error band (fill_between) or errorbars
    -> kwfill\tdictionary like **kwargs transmitted to pyplot.fill_between
    """

    # try if x and y are uncertainties.unumpy.uarrays
    try:
        x_n, x_s = separate_uarray(x)
    except AttributeError:
        x_n, x_s = array(x), 0  # type: ignore

    try:
        y_n, y_s = separate_uarray(y)
    except AttributeError:
        y_n, y_s = array(y), 0  # type: ignore

    if kwfill is None:
        kwfill = {}

    if band:
        kwargs.update({"linewidth": kwargs.get("linewidth", 1.0)})
        ret_plot = plot(x_n, y_n, *args, **kwargs)
        kwfill.update({"alpha": kwfill.get("alpha", 0.6)})
        ret_fill = fill_between(x=x_n, y1=y_n - y_s, y2=y_n + y_s, **kwfill)  # type: ignore
        return ret_plot, ret_fill
    else:
        kwargs.update({"capsize": kwargs.get("capsize", 2)})
        return errorbar(x_n, y_n, xerr=x_s, yerr=y_s, *args, **kwargs)


def separate_uarray(uarr: Iterable) -> tuple[ndarray, ndarray]:
    """Seperate an uncertainties.unumpy.uarray in n and s parts."""
    n = array([x.n for x in uarr])
    s = array([x.s for x in uarr])
    return n, s


def integrate(
    x: Iterable,
    y: Iterable,
    start: int = 0,
    stop: int = -1,
) -> float:
    """Numerically integrate a given array with specified boundaries

    Paramters:
    -> x\tarray of x-coordinates
    -> y\tarray of y-coordinates
    -> start=0\tlower integration boundary as index of array
    -> stop=-1\tupper integration boundary as index of array
    """
    return trapezoid(y[start:stop], x=x[start:stop])  # type: ignore

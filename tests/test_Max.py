from math import ceil

import uncertainties.core as uc

import labtool as lt
from labtool.monkeypatch_uncertainties import _digits_exponent_std_dev as _d_e_s_original


def _digits_exponent_std_dev(std_dev: float) -> tuple[int, int, float]:
    """Find the amount of significant digits and the exponent of those digits to the base 10.
    Also return the effective standard deviation.

    Provide data needed by function 'display' (subfunction 'EPM_precision') and
    function 'init' (subfunction 'round_conventional')

    Return one significant digit, except when this digit is 1, then return two.
    """

    if std_dev:  # std_dev != 0

        # exponent of base 10
        exponent = uc.first_digit(std_dev)
        print(f"{exponent=}")

        # calculate mantissa of std_dev
        # round to 3 digits to minimize machine epsilon
        mantissa = round(std_dev * 10 ** (-exponent), 3)
        print(f"{mantissa=}")

        # significant digits to consider for rounding
        sig_digits = 1

        # two significant digits if first digit is 1, one digit otherwise
        if mantissa <= 1.9:
            sig_digits += 1
            exponent -= 1
            mantissa *= 10
        elif mantissa > 9:
            sig_digits += 1

        # round up according to significant digits
        s = ceil(mantissa) * 10**exponent

        print("digits | exponent | std_dev")
        return sig_digits, exponent, s

    else:  # std_dev == 0
        return 0, 0, 0.0


def main() -> None:
    test_n = 3.22
    test_s = 0.091
    print(f"{test_s=}")
    print(_digits_exponent_std_dev(test_s))
    print(_d_e_s_original(test_s))

    test = lt.u.ufloat(test_n, test_s)
    print(test)

    # lt.monkeypatch_uncertainties.init()

    print(lt.u.ufloat(3.24323, 0.12).n)
    print(lt.u.ufloat(3.24323, 0.22).n)
    print(lt.u.ufloat(3.24323, 0.92).n)
    print(lt.u.ufloat(3.24323, 0.9).n)
    print(lt.u.ufloat(3.24323, 0.092).n)

    print("Example from Max:")
    print(lt.u.ufloat(1.911855e-10, 9.257050e-12))


if __name__ == "__main__":
    main()

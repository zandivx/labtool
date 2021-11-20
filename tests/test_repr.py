import labtool as lt

u = lt.u.ufloat(4, 0.2)

lst = [u**u]*5  # type: ignore


# print(u)  # check
# print(lst)  # check
# print(type(lst[0]))  # check
# print(lt.u.core.AffineScalarFunc.__repr__)  # works
s = lst[0].s
print(s)
print(lt.u.core.first_digit(s))
print(lt.u.ufloat_fromstr(str(lst[0])).s)

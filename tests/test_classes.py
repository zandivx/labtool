import labtool as lt


def fit():
    def func(x, a, b, c): return a * lt.np.exp(-b * x) + c

    xdata = lt.np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3, 0.5)
    rng = lt.np.random.default_rng()
    y_noise = 0.2 * rng.normal(size=xdata.size)
    ydata = y + y_noise
    #lt.plt.plot(xdata, ydata, 'b-', label='data')

    fit = lt.CurveFit(func, xdata, ydata)
    print(fit)
    # fit.save("saves/fit.csv")
    fit.plot(show=False)
    lt.plt.plot(lt.np.linspace(0, 4, 10)**2)
    # print(lt.plt.gcf())
    lt.plt.show()


# @lt.profile
def student():
    data = [28.89, 28.85, 28.92, 28.93, 28.98, 28.90, 28.85, 28.98, 28.88,
            28.91, 28.84, 28.86, 28.90, 28.87, 28.86, 28.91, 28.93, 28.86, 28.89, 28.89]

    series = lt.Student(data, 1)

    print(series)
    print(repr(series))
    series.save("saves/series.csv")


def interpolate():
    data = [28.89, 28.85, 28.92, 28.93, 28.98, 28.90, 28.85, 28.98, 28.88,
            28.91, 28.84, 28.86, 28.90, 28.87, 28.86, 28.91, 28.93, 28.86, 28.89, 28.89]

    interp = lt.Interpolation(range(len(data)), data)
    print(interp)
    interp.save("saves/interp.csv")
    interp.plot(style_in="o")


def student_array():
    lst = []
    x = 0
    for i in range(5):
        if i == 0:
            x = lt.pd.read_csv(f"data/{i}.csv", names=["x", "0"]).iloc[:, 0]
        lst.append(lt.pd.read_csv(
            f"data/{i}.csv", names=["x", f"{i}"]).iloc[:, 1])

    sa = lt.StudentArray(lst)
    print(lt.np.array(sa))
    print(sa.n)
    # lt.plt_uplot(x, sa, band=True)  # type: ignore
    # lt.plt.show()
    sa.plot(x)


# TODO:
def student_array_2():
    df_ges = lt.pd.DataFrame()
    for i in (0, 6):
        name = f"{i}_ref"
        df_ges["lambda"] = lt.pd.read_csv(
            f'data/3/{name}.csv').iloc[973:2729, 0]
        df_ges[name] = lt.pd.read_csv(f'data/3/{name}.csv').iloc[973:2729, 1]
    for i in range(1, 6):
        name = f"{i}_meth"
        df_ges[name] = lt.pd.read_csv(f'data/3/{name}.csv').iloc[973:2729, 1]

    print(df_ges)

    student_meth = df_ges.iloc[:, 3:]
    st_arr = lt.StudentArray(student_meth)
    print(st_arr)
    # st_arr.plot(df_ges["lambda"])


def student_nested():
    stud = lt.Student(range(23))
    lst = [stud]*5
    print(lst)


def main():
    student_array_2()


if __name__ == "__main__":
    main()

import numpy

from Cell2 import Cell
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from numba import jit
import random
import math
import csv
import scipy as sci

TRACK2PIX = 200


def get_data_from_csv(csv_file, x_cut):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    t = np.array([float(row[0]) for row in data])
    x = np.array([float(row[1]) for row in data])
    y = np.array([float(row[2]) for row in data])
    if x_cut != None:
        a = negate_drift(x[x_cut:])
        x = np.concatenate([x[0:x_cut], a])
    # y = negate_drift(y)
    return t, x, y


def negate_drift(x):
    n_array = np.linspace(0, len(x) - 1, len(x))
    fit = sci.stats.linregress(n_array, x)
    a = x - n_array * fit.slope
    return a


def spin_data(theta, x, y):
    xtag = x * math.cos(theta) - y * math.sin(theta)
    ytag = y * math.cos(theta) + x * math.sin(theta)
    return xtag, ytag


def return_average_spins(m, x, y, num_doors, size):
    temp = []
    for i in range(m):
        pass


def spin_array_and_get_time(theta, x, y, size, num_doors, epsilon):
    x_spun, y_spun = spin_data(theta, x, y)
    cell = Cell(size, [], x_spun, y_spun, True, num_doors, epsilon)
    maybe = cell.get_IOE()
    return maybe


def multiple_areas(n, start, end, x, y, num_doors):
    epsilon = 0.02
    areas = np.linspace(start, end, n)
    times = []
    err = []
    for i in tqdm(range(n)):
        temp = []
        cell = Cell(math.sqrt(areas[i]), [], x, y, True, num_doors, epsilon)
        temp = cell.get_all_exit_times(25)
        if temp != []:
            tt = np.array(temp)
            times.append(sum(tt) / len(tt))
            err.append(tt.std() / math.sqrt(len(tt)))
        else:
            break
    return np.array(times), np.array(areas[0:len(times)]), np.array(err)


def multiple_areas_t_coll(n, start, end, x, y, num_doors):
    epsilon = 0.02
    areas = np.linspace(start, end, n)
    times = []
    err = []
    for i in tqdm(range(n)):
        temp = []
        cell = Cell(math.sqrt(areas[i]), [], x, y, True, num_doors, epsilon)
        temp = cell.get_all_average_t_coll(25, 10)
        if temp != []:
            tt = np.array(temp)
            times.append(sum(tt) / len(tt))
            err.append(tt.std() / math.sqrt(len(tt)))
        else:
            break
    return np.array(times), np.array(areas[0:len(times)]), np.array(err)


def multiple_epsilon(n, start, end, x, y, num_doors, size):
    base = pow(end / start, 1 / n)
    epsilon_exp = np.linspace(1, n, n)
    epsilon = np.power(base, epsilon_exp) * start * base
    times = []
    err = []
    for i in tqdm(range(n)):
        cell = Cell(size, [], x, y, True, num_doors, epsilon[i], spin=random.uniform(0, 2 * math.pi))
        temp = cell.get_all_exit_times(25)
        if temp != []:
            tt = np.array(temp)
            times.append(sum(tt) / len(tt))
            err.append(tt.std() / math.sqrt(len(tt)))
        else:
            break
    return np.array(times), np.array(epsilon[0:len(times)]), np.array(err)


def multiple_doors(n, start, x, y, epsilon, size):
    num_doors = np.linspace(start, start + n - 1, n)
    num_doors = num_doors.astype(numpy.int64)
    times = []
    err = []
    for i in tqdm(range(n)):
        cell = Cell(size, [], x, y, True, num_doors[i], epsilon, spin=random.uniform(0, 2 * math.pi))
        temp = cell.get_all_exit_times(25)
        if temp != []:
            tt = np.array(temp)
            times.append(sum(tt) / len(tt))
            err.append(tt.std() / math.sqrt(len(tt)))
        else:
            break
    return np.array(times), np.array(num_doors[0:len(times)]), np.array(err)


def plot_graphs(xaxis, yaxis, yerr, title="", ylabel="", xlabel="", legend_entries=[]):
    """
    :param xaxis: array of x axis entries
    :param yaxis: array of y axis entries
    :param yerr: array of y error entries
    :param title: graph title
    :param ylabel: y axis title
    :param xlabel: x axis title
    :param legend_entries: array of legend titles
    :return:
    """
    leg = tuple(legend_entries)
    figure, axis = plt.subplots(1, 1)
    for i in range(len(xaxis)):
        axis.errorbar(xaxis[i], yaxis[i], yerr[i], fmt=".")
    figure.suptitle(title, fontsize=20)
    axis.set_ylabel(ylabel, fontsize=16)
    axis.set_xlabel(xlabel, fontsize=16)
    axis.legend(leg, loc=0)
    plt.show()


def usable_print(arr):
    print("[", end="")
    for i in range(len(arr)):
        print(arr[i], end=", ")
    print("]")


if __name__ == '__main__':

    if False:
        for i in range(3, 7):
            q = None
            alpha = 1
            if i == 2:
                q = 3500
                alpha = 1 / 200
            time, x_pos, y_pos = get_data_from_csv('Lab2_extension_csv_files/p' + str(i) + '.csv', q)
            if i == 2:
                y_pos = negate_drift(y_pos)

            if i == 4:
                alpha = 1 / 200
                x_pos = negate_drift(x_pos)
                y_pos = np.concatenate([y_pos[0:500], negate_drift(y_pos[500:3500]), negate_drift(y_pos[3500:])])
            if i == 5:
                a = negate_drift(x_pos[0:1000])
                b = (negate_drift(x_pos[1000:4000]) + a[999] - x_pos[1000])
                c = x_pos[4000:5250] + b[2999] - x_pos[4000]
                d = negate_drift(x_pos[5250:]) + c[1249] - x_pos[5250]
                x_pos = negate_drift(np.concatenate([a, b, c, d]))
                y_pos = np.concatenate([y_pos[0:300], negate_drift(y_pos[300:])])
            # plt.plot(x_pos)
            # plt.show()
            # plt.plot(y_pos)
            # plt.show()
            # plt.plot(x_pos, y_pos)
            # plt.show()
            print("p" + str(i))
            print()
            ex, why, err = multiple_epsilon(300, 0.4, 0.008, x_pos, y_pos, 4, 10 * alpha)
            print("\np" + str(i) + " multiple_epsilon")
            print("exit index", end=" = ")
            usable_print(ex)
            print("epsilons", end=" = ")
            usable_print(why)
            print("errors", end=" = ")
            usable_print(err)
            print()
            print()

            ex, why, err = multiple_areas(300, 25 * (alpha ** 2), 400 * (alpha ** 2), x_pos, y_pos, 4)
            print("\np" + str(i) + " multiple_areas")
            print("exit index", end=" = ")
            usable_print(ex)
            print("epsilons", end=" = ")
            usable_print(why)
            print("errors", end=" = ")
            usable_print(err)
            print()
            print()

            ex, why, err = multiple_doors(300, 1, x_pos, y_pos, 0.02, 10 * alpha)
            print("\np" + str(i) + " multiple_doors")
            print("exit index", end=" = ")
            usable_print(ex)
            print("epsilons", end=" = ")
            usable_print(why)
            print("errors", end=" = ")
            usable_print(err)
            print()
            print()
        # plt.plot(x_pos, y_pos)
        # cell = Cell((x_pos[0], y_pos[0]), (1,1), [((0.5,0), (0.6, 0))],x_pos, y_pos)
        # x = np.linspace(10, 0, 1000)
        # y = np.linspace(0, 5, 1000)
        # cell = Cell((x_pos[0], y_pos[0]), 25, [((25, 2), (25, 3))], x_pos, y_pos)
        # cell = Cell((x_pos[0], y_pos[0]), 25, [], x_pos, y_pos, True, 4, 1 / 100)
        # j = cell.get_IOE()
        # print(j)
        # cell.plot_movement()
        # ex, why = multiple_epsilon(100, 0.01, 0.0017, 40, x_pos, y_pos, 16, 5)
    time, x_pos, y_pos = get_data_from_csv('Lab2_extension_csv_files/p1.csv', None)

    cell = Cell(10, [], x_pos, y_pos, True, 4, 0.01, spin=random.uniform(0, 2 * math.pi))
    # ex, why = cell.get_mod_diff()
    # ex, why, err = multiple_areas_t_coll(50, 25, 4000, x_pos, y_pos, 4)
    # plt.errorbar(why, ex, err)
    # plt.show()
    ex = [3.004877143606618, 5.369170911733684, 6.417485345415311, 8.495268022150178, 9.274111104712143,
          9.643940064297073, 10.27069076963254, 11.457217319208834, 12.211503228033521, 11.7635205409172,
          13.04777776967352, 13.283900002118862, 13.514354338206768, 14.464417984590908, 14.629060765239592,
          15.073752559715114, 14.836817224255848, 15.561733555420536, 16.144560805133313, 16.724954955276754,
          17.014774177337493, 17.620704621899165, 17.67944520752128, 18.23124230511323, 18.405835541433795,
          18.356757269048543, 18.680060029349168, 19.204514175362156, 19.83689938699298, 20.172316702157467,
          20.549644480349635, 19.67092759006373, 20.2183591254764, 20.76403595607315, 20.658154705433617,
          20.726315689276923, 21.366326489114343, 21.458926824472368, 21.453303445609794, 22.701929226244555,
          22.956842523189668, 23.52796508369022, 24.34948323582567, 25.23094875410333, 25.691805953129233,
          26.402827900761146, 27.612323042580464, 27.171597368172097, 28.42612403865894, 27.88720892987894, ];
    why = [25.0, 106.12244897959184, 187.24489795918367, 268.36734693877554, 349.48979591836735, 430.61224489795916,
           511.734693877551, 592.8571428571429, 673.9795918367347, 755.1020408163265, 836.2244897959183,
           917.3469387755102,
           998.469387755102, 1079.591836734694, 1160.7142857142858, 1241.8367346938776, 1322.9591836734694,
           1404.0816326530612, 1485.204081632653, 1566.3265306122448, 1647.4489795918366, 1728.5714285714287,
           1809.6938775510205, 1890.8163265306123, 1971.938775510204, 2053.0612244897957, 2134.183673469388,
           2215.3061224489797, 2296.4285714285716, 2377.5510204081634, 2458.673469387755, 2539.795918367347,
           2620.918367346939, 2702.0408163265306, 2783.1632653061224, 2864.285714285714, 2945.408163265306,
           3026.530612244898,
           3107.6530612244896, 3188.7755102040815, 3269.8979591836733, 3351.0204081632655, 3432.1428571428573,
           3513.265306122449, 3594.387755102041, 3675.5102040816328, 3756.6326530612246, 3837.7551020408164,
           3918.877551020408, 4000.0, ];
    err = [0.34052377448775056, 0.607922819054442, 0.6733554968308091, 1.236578967280279, 0.8104365332595395,
           0.6394068855545584, 0.8430787799921038, 0.7389062776014941, 0.7529799475323716, 0.6749429054839317,
           0.7440382937388009, 0.6773712442160055, 0.7380689465144853, 0.720429544653018, 0.6747066279945023,
           0.6200837708994985, 0.5609455381638613, 0.8241285716033926, 0.8441891276537923, 0.8803282253689712,
           0.7412885908379159, 0.7838409861440366, 0.7148895364837127, 0.7468278683720719, 0.7642053146645311,
           0.7150126020195401, 0.7250369871883842, 0.6906160140498182, 0.8844448292688132, 0.9298444217582793,
           1.1662726952001556, 0.7175726311681286, 0.6518933127519808, 0.7660214709057284, 0.7441497016150135,
           1.2799607497909151, 1.3262954311543558, 1.3331455006280784, 1.3165418345451574, 1.375598194498881,
           1.032874928074838, 1.0285964520822015, 1.1079780421480372, 1.1041339480192203, 1.0083561255078815,
           1.0445326466657863, 1.2856046585476764, 1.0853893129536214, 1.5014861038781042, 1.4206441860822239, ]
    plt.errorbar(why, ex, err)
    plt.show()
    # usable_print(ex)
    # usable_print(why)
    # usable_print(err)
    # ex, why, err = multiple_epsilon(300, 0.2, 0.02, x_pos, y_pos, 4, 10)
    if True:
        # print(ex)
        # print(why)
        # plt.plot(why, ex)
        # plt.show()
        # plt.plot(why, np.log(ex))
        # plt.show()
        pass
    # usable_print(ex)
    # usable_print(why)
    # usable_print(err)
    # print()
    # print()
    # print()
    # plt.plot(why, ex)
    # plt.show()
    # ex, why, err = multiple_areas(300, 25, 400, x_pos, y_pos, 4)

    # plt.plot((why), ex)
    # plt.show()

    # a,b = cell.get_movement()
    # plt.plot(a[700:j],b[700:j])
    # plt.plot([25,25],[2,3])
    # plt.show()

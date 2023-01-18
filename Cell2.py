import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import math


class Cell:

    def __init__(self, dimension, door_list: list[tuple[tuple[float, float], tuple[float, float]]], x, y,
                 external_doors=False, number_of_doors=1, epsilon=0.1, spin=0):
        """
        :param center: irrelevant
        :param dimension: tuple(x_len, y_len)
        :param door_list: list of tuples of tuples
                for horizontal doors, door[0][0]<door[1][0]
                for vertical doors, door[0][1]<door[1][1]
        :param x: ndarray of x locations
        :param y: ndarray of y locations
        """
        x, y = self.spin_data(spin, x, y)
        self.center = (x[0], y[0])
        self.dimensions = (dimension, dimension)
        self.perimeter = 2 * self.dimensions[0] + 2 * self.dimensions[1]
        self.door_list = door_list
        self.modulo_list_x = np.linspace(0, 0, len(x))
        self.modulo_list_y = np.linspace(0, 0, len(y))
        if external_doors:
            self.door_list = self.door_list_generator(number_of_doors, epsilon)
        self.x = np.array([x[i] - x[0] + self.dimensions[0] / 2 for i in range(len(x))])
        self.y = np.array([y[i] - y[0] + self.dimensions[1] / 2 for i in range(len(x))])
        self.simulated_x = self.x
        self.simulated_y = self.y
        self.delta_list = np.zeros((len(x) - 1, 2))
        self.threshold = 0
        for i in range(1, len(x)):
            self.delta_list[i - 1][0] = self.x[i] - self.x[i - 1]
            self.delta_list[i - 1][1] = self.y[i] - self.y[i - 1]
            if self.threshold < max(abs(self.delta_list[i - 1][0]), abs(self.delta_list[i - 1][1])):
                self.threshold = max(abs(self.delta_list[i - 1][0]), abs(self.delta_list[i - 1][1]))
        self.modulate()

    @staticmethod
    def spin_data(theta, x, y):
        xtag = x * math.cos(theta) - y * math.sin(theta)
        ytag = y * math.cos(theta) + x * math.sin(theta)
        return xtag, ytag

    def door_list_generator(self, number_of_doors, epsilon):
        delta = (epsilon / number_of_doors) * self.perimeter
        p = [number_of_doors // 4] * 4
        for i in range(number_of_doors % 4):
            p[i] += 1
        door_list = []
        for i in range(4):
            if p[i] > 0:
                if p[i] > 1:
                    spread = np.linspace(self.dimensions[0] / 3 - delta / 2, 2 * self.dimensions[0] / 3 - delta / 2,
                                         p[i])
                else:
                    spread = np.linspace(self.dimensions[0] / 2 - delta, self.dimensions[0] / 2 - delta, p[i])
                for j in range(p[i]):
                    left_x = (1 - (i // 2)) * (spread[j]) + ((i // 2) * (i % 2)) * self.dimensions[1]
                    right_x = (1 - (i // 2)) * (spread[j] + delta) + ((i // 2) * (i % 2)) * self.dimensions[1]
                    left_y = (i // 2) * (spread[j]) + ((1 - (i // 2)) * (i % 2)) * self.dimensions[1]
                    right_y = (i // 2) * (spread[j] + delta) + ((1 - (i // 2)) * (i % 2)) * self.dimensions[1]
                    door_list.append(((left_x, left_y), (right_x, right_y)))
        return door_list

    def get_all_exit_times(self, delta):
        time = 0
        t = []
        while (time != None):
            time = self.get_IOE()
            # time = self.plot_movement()
            if len(self.x) > delta:
                self.create_new_point(delta)
            if time != None:
                t.append(time)
        return t

    def get_all_average_t_coll(self, delta, threshold):
        time = 0
        start = True
        t = []
        tt = []
        while (len(t) > threshold or start):
            t=[]
            start = False
            for i in range(1, len(self.x)):
                time += 1
                if self.modulo_list_x[i - 1] != self.modulo_list_x[i] or self.modulo_list_y[i - 1] != self.modulo_list_y[i]:
                    t.append(time)
                    time = 0
            ttt = np.array(t)
            tt.append(sum(ttt) / len(ttt))
            # time = self.plot_movement()
            if len(self.x) > delta:
                self.create_new_point(delta)
        return tt

    def create_new_point(self, delta):
        self.delta_list = self.delta_list[delta:]
        self.x = self.x[delta:] - self.x[delta] + self.dimensions[0] / 2
        self.y = self.y[delta:] - self.y[delta] + self.dimensions[1] / 2
        self.modulate()

    def modulate(self):
        self.modulo_list_x = np.array([round(self.x[i] // self.dimensions[0]) for i in range(len(self.x))])
        self.modulo_list_y = np.array([round(self.y[i] // self.dimensions[1]) for i in range(len(self.y))])
        self.simulated_x = self.x - (self.x // self.dimensions[0]) * self.dimensions[0]
        self.simulated_y = self.y - (self.y // self.dimensions[1]) * self.dimensions[1]

        for i in range(len(self.simulated_x)):
            if self.modulo_list_x[i] % 2 == 1:
                self.simulated_x[i] = self.dimensions[0] - self.simulated_x[i]
                if i != len(self.simulated_x) - 1:
                    self.delta_list[i][0] = -self.delta_list[i][0]
            if self.modulo_list_y[i] % 2 == 1:
                self.simulated_y[i] = self.dimensions[1] - self.simulated_y[i]
                if i != len(self.simulated_x) - 1:
                    self.delta_list[i][1] = -self.delta_list[i][1]

    def check_door(self, i):
        if i == 232:
            pass
        if self.modulo_list_x[i - 1] != self.modulo_list_x[i]:
            for door in self.door_list:
                # if the door is vertical
                if door[0][0] == door[1][0]:
                    x_delta = self.delta_list[i - 1][0]
                    y_delta = self.delta_list[i - 1][1]
                    if x_delta > 0:
                        y_hit_wall = self.simulated_y[i - 1] + (
                                (self.dimensions[0] - self.simulated_x[i - 1]) / x_delta) * y_delta
                        if door[1][1] > y_hit_wall > door[0][1] and abs(
                                self.simulated_x[i - 1] - door[0][0]) <= abs(x_delta):
                            return True
                    elif x_delta < 0:
                        y_hit_wall = self.simulated_y[i - 1] - (self.simulated_x[i - 1] / x_delta) * y_delta
                        if door[1][1] > y_hit_wall > door[0][1] and abs(
                                self.simulated_x[i - 1] - door[0][0]) < abs(x_delta):
                            return True

        elif self.modulo_list_y[i - 1] != self.modulo_list_y[i]:
            for door in self.door_list:
                # if the door is horizontal
                if door[0][1] == door[1][1]:
                    x_loc = self.simulated_x[i - 1]
                    y_loc = self.simulated_y[i - 1]
                    x_delta = self.delta_list[i - 1][0]
                    y_delta = self.delta_list[i - 1][1]
                    if y_delta > 0:
                        x_hit_wall = self.simulated_x[i - 1] + (
                                (self.dimensions[1] - self.simulated_y[i - 1]) / y_delta) * x_delta
                        if door[1][0] > x_hit_wall > door[0][0] and abs(
                                self.simulated_y[i - 1] - door[0][1]) < abs(y_delta):
                            return True
                    elif y_delta < 0:
                        x_hit_wall = self.simulated_x[i - 1] - (self.simulated_y[i - 1] / y_delta) * x_delta
                        if door[1][0] > x_hit_wall > door[0][0] and abs(
                                self.simulated_y[i - 1] - door[0][1]) < abs(y_delta):
                            return True
        return False

    def get_IOE(self):
        for i in range(1, len(self.x)):
            if self.check_door(i):
                return i
        return None

    def get_movement(self):
        return self.simulated_x, self.simulated_y

    def plot_movement(self, starting_index=0):
        j = self.get_IOE()
        # print(j)
        a, b = self.get_movement()
        a = a[0:j]
        b = b[0:j]
        new_x = a[j - 1] + self.delta_list[j - 1][0]
        new_y = b[j - 1] + self.delta_list[j - 1][1]
        atag = np.append(a, new_x)
        btag = np.append(b, new_y)
        plt.plot(atag[starting_index:len(atag)], btag[starting_index:len(btag)])
        for door in self.door_list:
            plt.plot([door[0][0], door[1][0]], [door[0][1], door[1][1]])
        plt.show()
        return j

    def get_mod_diff(self):
        for i in range(len(self.modulo_list_x)):
            # print("(" + str(self.modulo_list_x[i]) + "," + str(self.modulo_list_y[i]) + ")")
            if i > 0:
                print("(" + str(self.modulo_list_x[i] - self.modulo_list_x[i - 1]) + "," + str(
                    self.modulo_list_y[i] - self.modulo_list_y[i - 1]) + ")")
        return np.array([((self.modulo_list_x[i] - self.modulo_list_x[i - 1]) for i in
                          range(len(self.modulo_list_x) - 1))]), np.array(
            [((self.modulo_list_y[i] - self.modulo_list_y[i - 1]) for i in range(len(self.modulo_list_x) - 1))])

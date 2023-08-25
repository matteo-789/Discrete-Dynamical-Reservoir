import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression, Ridge
import math
from reservoirpy.observables import rmse
from sklearn.base import BaseEstimator, ClassifierMixin
import itertools
from enum import Enum

S3_combinations = list(itertools.product(range(3), repeat=3))

def rule_30_3S(neighbors):
    if (
        neighbors == list(S3_combinations[0])
        or neighbors == list(S3_combinations[8])
        or neighbors == list(S3_combinations[9])
        or neighbors == list(S3_combinations[16])
        or neighbors == list(S3_combinations[18])
        or neighbors == list(S3_combinations[19])
        or neighbors == list(S3_combinations[25])
    ):
        return 0
    elif (
        neighbors == list(S3_combinations[4])
        or neighbors == list(S3_combinations[5])
        or neighbors == list(S3_combinations[13])
        or neighbors == list(S3_combinations[14])
        or neighbors == list(S3_combinations[22])
        or neighbors == list(S3_combinations[24])
    ):
        return 2
    else:
        return 1


def rule_110_3S(neighbors):
    if (
        neighbors == list(S3_combinations[0])
        or neighbors == list(S3_combinations[4])
        or neighbors == list(S3_combinations[8])
        or neighbors == list(S3_combinations[11])
        or neighbors == list(S3_combinations[12])
        or neighbors == list(S3_combinations[16])
        or neighbors == list(S3_combinations[20])
        or neighbors == list(S3_combinations[21])
        or neighbors == list(S3_combinations[25])
    ):
        return 0
    else:
        return 1


class CML_map(Enum):
    logistic = 1
    sig = 2
    tanh = 3
    circle = 4


class Coupling_types(Enum):
    basic = 1
    double = 2


class ReservoirType(Enum):
    BECA = 1
    MECA = 2
    CML = 3


class TypeRule(Enum):
    dual = 1
    uniform = 2
    random = 3

class CAReservoir:
    def __init__(self, careservoir, leaking_rate=1):
        self.Reservoir = careservoir
        self.Readout = Readout(leaking_rate)

    def fit(self, data, target):
        self.Reservoir.run_CA(data, target, self.Readout, predict=False)
        self.Readout.fit(self.Reservoir.get_output(), target)
        return None

    def predict(self, data):
        self.Reservoir.step = 1
        self.Reservoir.run_CA(data, None, None, predict=True)
        predictions = self.Readout.predict(self.Reservoir.get_output())
        return predictions

    def score(self, X, Y):
        return rmse(Y, self.predict(X))


class Reservoir(CAReservoir):
    """
    This class is used for execution it is composed of 4 functions: init, fit, predict and score, and some utils
    __init__:
    tt_step: input recall, defines number of iterations before input recall
    map_lattices: map to use for CML reservoir
    rule: rule to use for Binary ECA reservoir
    r:
    coupling_s: coupling strength coefficient which defines how much of the past state we keep
    iteration: number of generation
    CA_nb: or C-paramter number of grid we concatenate
    replication: number of time we cast input in grid randomly
    project_type: type of reservoir
    rule_type: dual, simple, uniform or random, defines rule type for CA or pinning type for CML
    coupling_type: defines if we use a coupling equation for one or two neighbors
    leaking_rate: paramter for readout layer to control dynamics in this one (Ridge regression)
    """

    def __init__(
        self,
        size="30",
        coupling_type: Coupling_types = Coupling_types.basic,
        rule=4,
        iteration=50,
        CA_nb=1,
        replication=1,
        tt_step=3,
        project_type: ReservoirType = ReservoirType.MECA,
        rule_type: TypeRule = TypeRule.dual,
        coupling_s=0.7,
        r=3.9,
        p=0.45,
        map_lattices: CML_map = CML_map.logistic,
        use="optimize",
    ):
        self.rule = rule
        self.map_lattices = map_lattices
        self.iteration = iteration
        self.n_bits = {
            "111": "0",
            "110": "0",
            "101": "0",
            "100": "1",
            "011": "1",
            "010": "1",
            "001": "1",
            "000": "0",
        }
        self.project_type = project_type
        self.coupling_type = coupling_type
        self.replication = int(replication)
        self.size = size
        self.tt_step = tt_step
        self.rule_type = rule_type
        self.CA_nb = int(CA_nb)
        self.new_grid = np.zeros(
            self.replication * self.iteration * self.CA_nb, dtype=float
        )
        self.grid = np.zeros(
            self.replication * self.iteration * self.CA_nb, dtype=float
        )
        self.coupling_strength = coupling_s
        self.r = r
        self.p = p

    def get_output(self):
        return self.final_grid

    def generate_nonlinear_rule(self, rule_number):
        """
        Generate a nonlinear rule for start env in grid
        """
        rule = np.zeros(8, dtype=int)
        binary_rule = np.binary_repr(rule_number, width=8)
        for i in range(8):
            rule[i] = int(binary_rule[i])
        return rule

    def set_multiple_rules(self):
        self.n_bits1 = {
            "111": "0",
            "110": "0",
            "101": "0",
            "100": "1",
            "011": "1",
            "010": "1",
            "001": "1",
            "000": "0",
        }
        self.n_bits2 = {
            "111": "0",
            "110": "0",
            "101": "0",
            "100": "1",
            "011": "1",
            "010": "1",
            "001": "1",
            "000": "0",
        }
        binary_rule = format(self.rule[0], "08b")
        for i, val in enumerate(self.n_bits1):
            self.n_bits1[val] = binary_rule[i]
        binary_rule = format(self.rule[1], "08b")
        for i, val in enumerate(self.n_bits2):
            self.n_bits2[val] = binary_rule[i]

    def nonlinear_env(self, neighbors, binary_data, j):
        """
        Set nonlinear env to enable nonlinear dynamics in reservoir
        """
        rule = self.generate_nonlinear_rule(self.rule[0])
        test = np.zeros(len(binary_data[0]), dtype=int)
        for h in range(len(binary_data[j])):
            neighbors.append(binary_data[j][(h - 1) % len(binary_data[0])])
            neighbors.append(binary_data[j][h])
            neighbors.append(binary_data[j][(h + 1) % len(binary_data[0])])
            test[h] = rule[int(sum(neighbors))]
            neighbors = []
        return test

    def run_CA(self, data, target, readout, threshold=0.5, predict=False, first_arr=1):
        """
        Function divided in 3 parts: Binary ECA, 3 states ECA, CML
        Main part of the fit function
        """
        neighbors = []
        if self.project_type is ReservoirType.BECA:
            self.set_multiple_rules()
            binary_data = np.where(data >= np.mean(data), 1, 0)
            size_CA = self.iteration * self.replication * self.CA_nb
            neighbors = []
            grid_IR = np.zeros(self.iteration * len(data[0]), dtype=int)
            self.new_grid = np.zeros(size_CA, dtype=int)
            self.grid = np.zeros(size_CA, dtype=int)
            sequence = np.array([])

            if predict == True:
                self.grid = self.get_output()[-1][:size_CA]
                first_arr = 0
            self.final_grid = []

            for j in range(len(binary_data)):
                binary_data[j] = self.nonlinear_env(neighbors, binary_data, j)
                self._replication(binary_data, j, int)

                for e in range(self.iteration):
                    self.new_grid = np.zeros(size_CA, dtype=int)
                    if e % self.tt_step == 0:
                        random_indices = np.random.choice(
                            size_CA, size=len(binary_data[j]), replace=False
                        )
                        self.new_grid[random_indices] = binary_data[j]
                    for i in range(size_CA):
                        neighbors = self.get_neighborhood(i, size_CA)
                        if i < int(size_CA / 2):
                            self.apply_rule(i, neighbors, self.n_bits1)
                        else:
                            self.apply_rule(i, neighbors, self.n_bits2)
                    self.grid = self.new_grid
                    sequence = np.append(sequence, self.grid)
                self.final_grid.append(sequence.flatten())
                sequence = np.array([])

        elif self.project_type is ReservoirType.CML:
            size_CA = self.iteration * self.replication * self.CA_nb
            neighbors = []
            self.new_grid = np.zeros(size_CA, dtype=float)
            self.grid = np.zeros(size_CA, dtype=float)
            sequence = np.array([])

            self.final_grid = []

            for j in range(len(data)):
                self._replication(data, j, float)

                for e in range(self.iteration):
                    self.new_grid = np.zeros(size_CA, dtype=float)
                    if self.rule_type is TypeRule.random:
                        cases = np.random.choice(
                            size_CA, size=int(size_CA / 2), replace=False
                        )
                    elif self.rule_type is TypeRule.uniform:
                        cases = np.arange(1, size_CA, 2)
                    for i in range(size_CA):
                        neighbors = self.get_neighborhood(i, size_CA)
                        if i in cases:
                            self.new_grid[i] = self.grid[i]
                        else:
                            if self.coupling_type is Coupling_types.basic:
                                self.new_grid[
                                    i
                                ] = self.coupling_strength * self.mapping_function_CML(
                                    self.r, neighbors[1], e, i
                                ) + (
                                    1 - self.coupling_strength
                                ) * self.mapping_function_CML(
                                    self.r, neighbors[0], e, i
                                )
                            elif self.coupling_type is Coupling_types.double:
                                self.new_grid[i] = (
                                    1 - self.coupling_strength
                                ) * self.mapping_function_CML(
                                    self.r, neighbors[1], e, i
                                ) + (
                                    self.coupling_strength / 2
                                ) * (
                                    self.mapping_function_CML(
                                        self.r, neighbors[0], e, i
                                    )
                                    + self.mapping_function_CML(
                                        self.r, neighbors[2], e, i
                                    )
                                )

                    self.grid = self.new_grid
                    sequence = np.append(sequence, self.grid)
                self.final_grid.append(sequence.flatten())
                sequence = np.array([])

        elif self.project_type is ReservoirType.MECA:
            lower_threshold = 0.33
            upper_threshold = 0.66

            binary_data = np.zeros_like(data, dtype=int)
            binary_data[data < lower_threshold] = 0
            binary_data[(data >= lower_threshold) & (data <= upper_threshold)] = 1
            binary_data[data > upper_threshold] = 2
            size_CA = self.iteration * self.replication * self.CA_nb
            neighbors = []
            grid_IR = np.zeros(self.iteration * len(data[0]), dtype=int)
            self.new_grid = np.zeros(size_CA, dtype=int)
            self.grid = np.zeros(size_CA, dtype=int)
            sequence = np.array([])

            if predict == True:
                self.grid = self.get_output()[-1][:size_CA]
                first_arr = 0
            self.final_grid = []

            for j in range(len(binary_data)):
                self._replication(binary_data, j, int)

                for e in range(self.iteration):
                    self.new_grid = np.zeros(size_CA, dtype=int)
                    if e % self.tt_step == 0 and j != 0:
                        random_indices = np.random.choice(
                            size_CA, size=len(binary_data[j]), replace=False
                        )
                        self.new_grid[random_indices] = binary_data[j - 1]
                    for i in range(size_CA):
                        neighbors = self.get_neighborhood(i, size_CA)
                        if self.size == "30":
                            self.new_grid[i] = rule_30_3S(neighbors)
                        elif self.size == "110":
                            self.new_grid[i] = rule_110_3S(neighbors)
                    self.grid = self.new_grid
                    sequence = np.append(sequence, self.grid)
                self.final_grid.append(sequence.flatten())
                sequence = np.array([])

    def mapping_function_CML(self, r, x, e, i):
        if self.map_lattices is CML_map.logistic:
            return r * x * (1 - x)
        elif self.map_lattices is CML_map.circle:
            return (
                self.grid[e - 1, i]
                + self.r
                - 0.6 / (2 * np.pi) * np.sin(2 * np.pi * self.grid[e - 1, i])
            )
        elif self.map_lattices is CML_map.tanh:
            return np.tanh(r * x)
        elif self.map_lattices is CML_map.sig:
            return 1 - r * (x * x)

    def apply_rule(self, i, neighbors, wolfram_rule):
        neighbors = "".join(str(x) for x in neighbors)
        if str(neighbors) in wolfram_rule:
            self.new_grid[i] = wolfram_rule[str(neighbors)]

    def get_neighborhood(self, i, size):
        neighbors = []
        neighbors.append(self.grid[(i - 1) % size])
        neighbors.append(self.grid[i])
        neighbors.append(self.grid[(i + 1) % size])
        return neighbors

    def transition_function(self, final):
        result = self.grid + final
        result[result == 1] = np.random.randint(0, 2, size=result[result == 1].shape)
        result[result == 2] = 1
        result[result != 1] = 0
        return result

    def _replication(self, binary_data, j, dtype):
        tmp_ca = np.zeros(self.iteration * self.CA_nb * self.replication, dtype=dtype)
        for i in range(self.replication):
            random_indices = np.random.choice(
                int(len(tmp_ca) / self.replication),
                size=len(binary_data[j]),
                replace=False,
            )
            random_indices = random_indices + int(len(tmp_ca) / self.replication) * i
            tmp_ca[random_indices] = binary_data[j]
        if self.project_type is ReservoirType.BECA:
            self.grid = self.transition_function(tmp_ca)
        if (
            self.project_type is ReservoirType.CML
            or self.project_type is ReservoirType.MECA
        ):
            self.grid = tmp_ca


class Readout(CAReservoir):
    def __init__(self, leaking_rate=1.0):
        self.model = Ridge(alpha=leaking_rate)

    def fit(self, train, target):
        self.model.fit(train, target)

    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions

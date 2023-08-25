# Discrete-Dynamical-Reservoir
Python library to build and use discrete dynamical systems within recurrent neural networks.
This library is a user-friendly tool made to use discrete dynamical systems such as Binary ECA, 3 states ECA and CML as a reservoir.
Each type of reservoir has its hyper-parameters to enhance the reservoir performance.

This library works for Python 3.8 and higher.

## Quick example of how to code a CML reservoir
```python
import Discrete_Dynamical_Reservoir as DDSR


X_train = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[12,14,15]]
Y_train = [1,2,3,4,5]
X_test = [[6,4,4],[11,11,15],[1,2,4]]
Y_test = [2,5,1]


reservoir = DDSR.Reservoir(
                rule=(195, 60),
                replication=4,
                iteration=8,
                CA_nb=4,
                tt_step=2,
                project_type=DDSR.ReservoirType.BECA,
                rule_type=DDSR.TypeRule.dual,
                use="predict",
            )
CARe = DDSR.CAReservoir(reservoir, leaking_rate=0.076)

print("Model training...")
CARe.fit(X_train, Y_train)
print("Model predicting...")
print(CARe.score(X_test, Y_test))
```

## Installation
```
pip install disc
```

## Paper linked to this project

This project is part of my MSc project, the dissertation link to this code is available in the repository.

import numpy as np
import pandas as pd


def load_sonar():
    data = pd.read_csv('sonar.all-data',header=None)
    # https://www.simonwenkel.com/2018/08/23/revisiting_ml_sonar_mines_vs_rocks.html


    # identify sonar column names
    data.columns = ['X0','X1','X2','X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
                'X10', 'X11','X12','X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19',
                'X20','X21','X22','X23', 'X24', 'X25', 'X26', 'X27', 'X28', 'X29',
                'X30','X31','X32','X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39',
                'X40','X41','X42','X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49',
                'X50','X51','X52','X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'Class']

    data['Class'] = np.where(data['Class'] == 'R',0,1) #Change the Class representation

    # shuffle the data rows
    data = data.reindex(np.random.RandomState(seed=42).permutation(data.index))

    X = data.drop('Class',axis=1)
    y = data['Class']

    print(y.head())

    return X, y

load_sonar()
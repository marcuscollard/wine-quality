import os
import numpy as np
import pandas as pd
import hopsworks


INDEX = [3, 4, 5, 6, 7, 8, 9]
COLUMNS = ['fixed_acidity', 'volatile_acidity', 'citric_acid',
           'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
           'total_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol']
MIN_CONSTRAINTS = [
    [4.2, 0.17, 0.0, 0.7, 0.022, 3.0, 9.0, 0.9911, 2.87, 0.28, 8.0],
    [4.6, 0.11, 0.0, 0.7, 0.013, 3.0, 7.0, 0.9892, 2.74, 0.25, 8.4],
    [4.5, 0.1, 0.0, 0.6, 0.009, 2.0, 6.0, 0.98722, 2.79, 0.27, 8.0],
    [3.8, 0.08, 0.0, 0.7, 0.015, 1.0, 6.0, 0.98758, 2.72, 0.23, 8.4],
    [4.2, 0.08, 0.0, 0.9, 0.012, 3.0, 7.0, 0.98711, 2.84, 0.22, 8.6],
    [3.9, 0.12, 0.03, 0.8, 0.014, 3.0, 12.0, 0.98713, 2.88, 0.25, 8.5],
    [6.6, 0.24, 0.29, 1.6, 0.018, 24.0, 85.0, 0.98965, 3.2, 0.36, 10.4]
]
MAX_CONSTRAINTS = [
    [11.8, 1.58, 0.66, 16.2, 0.267, 289.0, 440.0, 1.0008, 3.63, 0.86, 12.6],
    [12.5, 1.13, 1.0, 17.55, 0.61, 138.5, 272.0, 1.001, 3.9, 2.0, 13.5],
    [15.9, 1.33, 1.0, 23.5, 0.611, 131.0, 344.0, 1.00315, 3.79, 1.98, 14.9],
    [14.3, 1.04, 1.66, 65.8, 0.415, 112.0, 294.0, 1.03898, 4.01, 1.95, 14.0],
    [15.6, 0.915, 0.76, 19.25, 0.358, 108.0, 289.0, 1.0032, 3.82, 1.36, 14.2],
    [12.6, 0.85, 0.74, 14.8, 0.121, 105.0, 212.5, 1.0006, 3.72, 1.1, 14.0],
    [9.1, 0.36, 0.49, 10.6, 0.035, 57.0, 139.0, 0.997, 3.41, 0.61, 12.9]
]


def generate_wine(quality, min_constraints, max_constraints):
    """Returns a single wine as a single row in a DataFrame"""

    columns = min_constraints.columns
    wine_df = pd.DataFrame(np.random.uniform(min_constraints.loc[quality], 
                                             max_constraints.loc[quality],
                                             size=(1, len(columns))), 
                           columns=columns)
    wine_df['type'] = np.random.choice(['white', 'red'])
    wine_df['quality'] = quality
    return wine_df


def get_random_wine():
    """Returns a DataFrame containing one random wine"""

    min_constraints = pd.DataFrame(MIN_CONSTRAINTS, index=INDEX, columns=COLUMNS)
    max_constraints = pd.DataFrame(MAX_CONSTRAINTS, index=INDEX, columns=COLUMNS)

    quality = np.random.randint(3, 10)
    wine_df = generate_wine(quality, min_constraints, max_constraints)

    return wine_df


def add_wine():
    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    wine_fg = fs.get_feature_group(name="wine", version=1)
    wine_fg.insert(wine_df)


if __name__ == "__main__":
    add_wine()

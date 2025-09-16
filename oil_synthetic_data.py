import numpy as np
import pandas as pd

rng = np.random.default_rng(50)

n_samples = 1000

data = {
    'Sample ID' : ['S' + str(i).zfill(3) for i in range(n_samples)],
    'density': rng.uniform(0.8, 1.0, n_samples),
    'viscosity_40C': rng.uniform(10, 1000, n_samples),
    'viscosity_100C': rng.uniform(5, 100, n_samples),
    'viscosity_index': rng.uniform(80, 200, n_samples),
    'flash_point': rng.uniform(150, 300, n_samples),
    'pour_point': rng.uniform(-40, 0, n_samples),
    'sulfur_content': rng.uniform(0.01, 5, n_samples),
    'api_gravity': rng.uniform(10, 40, n_samples),
    'thermal_conductivity': rng.uniform(0.1, 0.3, n_samples),
    'lubricity': rng.uniform(0.5, 1.5, n_samples),
    'carbon_residue': rng.uniform(0.1, 10, n_samples),
    'oxidation_stability': rng.uniform(100, 1000, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('oil synthetic dataset.csv', index = False)

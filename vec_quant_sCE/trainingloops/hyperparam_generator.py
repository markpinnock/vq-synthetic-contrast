import numpy as np


np.random.seed()
g_eta = np.round(np.power(10, np.random.uniform(np.log10(1e-3), np.log10(1e-5))), 5)
d_eta = np.round(np.power(10, np.random.uniform(np.log10(g_eta), np.log10(1e-5))), 5)
g_ch = np.random.choice([2 ** i for i in range(2, 7)])
d_ch = np.random.choice([2 ** i for i in range(2, np.log2(g_ch).astype("int32") + 1)])
g_layers = np.random.choice(list(range(6, 9)))
d_layers = np.random.choice(list(range(1, np.min([g_layers, 7]))))
lambda_ = np.round(np.power(10, np.random.uniform(np.log10(50), np.log10(10000))), 0).astype("int32")
mu = np.round(np.random.uniform(0.0, 0.7), 2)
Nz = np.random.choice([2 ** i for i in range(2, 7)])

d = {"d_eta": d_eta,
     "g_eta": g_eta,
     "d_ch": d_ch,
     "g_ch": g_ch,
     "d_layers": d_layers,
     "g_layers": g_layers,
     "lambda": lambda_,
     "mu": mu,
     "Nz": Nz}

print(d)
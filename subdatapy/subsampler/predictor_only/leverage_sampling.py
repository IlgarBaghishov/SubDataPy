import os, random, time, resource
from ase.geometry import get_distances
from copy import deepcopy
import numpy as np
from scipy.linalg import lstsq
import pandas as pd
from sklearn.linear_model import SGDRegressor

def load_files(file_name_structures, file_name_energies):
    df_structures = pd.read_hdf(file_name_structures)
    df_structures.sort_index(inplace=True)

    df_energies = pd.read_hdf(file_name_energies)
    df_energies.sort_values(by=["index"], inplace=True)

    df_structures = df_structures[df_structures.index.isin(df_energies["index"].values)]
    return df_structures, df_energies

file_name_structures = "/vast/home/baghishov/old_entropy/lanl_data/Be_large_subset_structures.h5"
file_name_energies = "/vast/home/baghishov/old_entropy/lanl_data/Be_low_3.hdf"
atoms_clmn = "ASEatoms_rescale"
df_structures, df_energies = load_files(file_name_structures, file_name_energies)
configs_num = df_structures[atoms_clmn].shape[0]

del df_energies

aw = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/aw.npy")
bw_low = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/bw_low.npy")
bw_high = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/bw_high.npy")
energy_selector = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/energy_selector.npy")
force_selector = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/force_selector.npy")
coeffs = np.load("/vast/home/baghishov/old_entropy/qSNAP/fidelity/Be/lowBe_reg/coefficients.npy")
df = pd.read_csv("df_diag.csv",index_col=0)
probabilities = df["e_diag"].values/df["e_diag"].sum()
slctd_num = 100

for _ in range(50):
    slctd = np.random.choice(df.index,slctd_num,replace=False,p=probabilities)
    unslctd = [i for i in df.index if i not in slctd]
    # slctd = df.sort_values(by="e_diag",ascending=False).index[:slctd_num]
    # unslctd = df.sort_values(by="e_diag",ascending=False).index[slctd_num:]
    print("Gathered all data")

    last_index = 0
    configs_index = []
    for i in range(configs_num):
        configs_index.append([last_index+j for j in range(1+3*len(df_structures[atoms_clmn].values[i]))])
        last_index += 1+3*len(df_structures[atoms_clmn].values[i])
    ind_configs_index = [i for i in range(configs_num)]
    random.seed(58)
    random.shuffle(ind_configs_index)
    hlfpnt = len(ind_configs_index)//2
    hlfpnt1 = [item for sublist_index in ind_configs_index[:hlfpnt] for item in configs_index[sublist_index]]
    hlfpnt2 = [item for sublist_index in ind_configs_index[hlfpnt:] for item in configs_index[sublist_index]]
    hlfpnt1_low = [item for unslctd_ind in unslctd for item in configs_index[ind_configs_index[unslctd_ind]]]
    hlfpnt1_high = [item for slctd_ind in slctd for item in configs_index[ind_configs_index[slctd_ind]]]

    print("\nFitting to selected "+str(slctd_num)+" high precision configurations and remaining low precision configurations")
    start_time = time.time()
    coefficients, *_ = lstsq(aw[hlfpnt1_high], bw_high[hlfpnt1_high], 1.0e-13)
    print("Fitting finished in", time.time()-start_time, "sec")
    train_residual = np.square(np.dot(aw[hlfpnt1],coefficients) - bw_high[hlfpnt1])
    print("Energy training RMSE is", np.sqrt(np.sum(train_residual*energy_selector[hlfpnt1]/22500)/energy_selector[hlfpnt1].sum()))
    print("Force training RMSE is", np.sqrt(np.sum(train_residual*force_selector[hlfpnt1])/force_selector[hlfpnt1].sum()))
    train_residual = np.square(np.dot(aw[hlfpnt1_high],coefficients) - bw_high[hlfpnt1_high])
    print("Energy training selected high precision points RMSE is", np.sqrt(np.sum(train_residual*energy_selector[hlfpnt1_high]/22500)/energy_selector[hlfpnt1_high].sum()))
    print("Force training selected high precision points RMSE is", np.sqrt(np.sum(train_residual*force_selector[hlfpnt1_high])/force_selector[hlfpnt1_high].sum()))
    test_residual = np.square(np.dot(aw[hlfpnt2],coefficients) - bw_high[hlfpnt2])
    print("Energy testing RMSE is", np.sqrt(np.sum(test_residual*energy_selector[hlfpnt2]/22500)/energy_selector[hlfpnt2].sum()))
    print("Force testing RMSE is", np.sqrt(np.sum(test_residual*force_selector[hlfpnt2])/force_selector[hlfpnt2].sum()))
    entire_test_diff = np.dot(aw,coefficients) - bw_high
    entire_test_residual = np.square(entire_test_diff)
    print("Energy RMSE if tested on the entire dataset is", np.sqrt(np.sum(entire_test_residual*energy_selector/22500)/energy_selector.sum()))
    print("Force RMSE if tested on the entire dataset is", np.sqrt(np.sum(entire_test_residual*force_selector)/force_selector.sum()))

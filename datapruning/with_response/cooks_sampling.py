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
file_name_energies = "/vast/home/baghishov/old_entropy/lanl_data/Be_lower_low_3.hdf"
atoms_clmn = "ASEatoms_rescale"
df_structures, df_energies = load_files(file_name_structures, file_name_energies)
configs_num = df_structures[atoms_clmn].shape[0]

del df_energies

aw = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/aw.npy")
bw_low = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/bw_lower.npy")
bw_high = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/bw_high.npy")
energy_selector = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/energy_selector.npy")
force_selector = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/force_selector.npy")
# coeffs = np.load("/vast/home/baghishov/old_entropy/qSNAP/fidelity/Be/lowerBe_reg/coefficients.npy")
df = pd.read_csv("../df_cooks.csv",index_col=0)
probabilities = df["e_cooks"].values/df["e_cooks"].sum()
slctd_num = 100
slctd_num_low = 2000

for _ in range(50):
    slctd = np.random.choice(df.index,slctd_num,replace=False,p=probabilities)
    slctd_low = df.sort_values(by="e_cooks",ascending=False).index[-slctd_num_low:]
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
    hlfpnt1_low = [item for slctd_low_ind in slctd_low for item in configs_index[ind_configs_index[slctd_low_ind]]]
    hlfpnt1_high = [item for slctd_ind in slctd for item in configs_index[ind_configs_index[slctd_ind]]]

    print("\nFitting a MLIP to unselected configurations only at low precision")
    coeffs, *_ = lstsq(aw[hlfpnt1_low], bw_low[hlfpnt1_low], 1.0e-13)
    # train_residual = np.square(np.dot(aw[hlfpnt1],coeffs) - bw_high[hlfpnt1])
    # print("Energy training RMSE is", np.sqrt(np.sum(train_residual*energy_selector[hlfpnt1]/22500)/energy_selector[hlfpnt1].sum()))
    # print("Force training RMSE is", np.sqrt(np.sum(train_residual*force_selector[hlfpnt1])/force_selector[hlfpnt1].sum()))
    # train_residual = np.square(np.dot(aw[hlfpnt1_high],coeffs) - bw_high[hlfpnt1_high])
    # print("Energy training selected high precision points RMSE is", np.sqrt(np.sum(train_residual*energy_selector[hlfpnt1_high]/22500)/energy_selector[hlfpnt1_high].sum()))
    # print("Force training selected high precision points RMSE is", np.sqrt(np.sum(train_residual*force_selector[hlfpnt1_high])/force_selector[hlfpnt1_high].sum()))
    # test_residual = np.square(np.dot(aw[hlfpnt2],coeffs) - bw_high[hlfpnt2])
    # print("Energy testing RMSE is", np.sqrt(np.sum(test_residual*energy_selector[hlfpnt2]/22500)/energy_selector[hlfpnt2].sum()))
    # print("Force testing RMSE is", np.sqrt(np.sum(test_residual*force_selector[hlfpnt2])/force_selector[hlfpnt2].sum()))
    # entire_test_diff = np.dot(aw,coeffs) - bw_high
    # entire_test_residual = np.square(entire_test_diff)
    # print("Energy RMSE if tested on the entire dataset is", np.sqrt(np.sum(entire_test_residual*energy_selector/22500)/energy_selector.sum()))
    # print("Force RMSE if tested on the entire dataset is", np.sqrt(np.sum(entire_test_residual*force_selector)/force_selector.sum()))

    print("\nFitting to selected "+str(slctd_num)+" high precision configurations and unselected low precision configurations")
    lamda = 1000000
    start_time = time.time()
    print("slctd_num_low is", slctd_num_low)
    coefficients, *_ = lstsq(np.vstack([np.sqrt(lamda)*np.eye(aw.shape[1]),aw[hlfpnt1_high]]),
                             np.concatenate([np.sqrt(lamda)*coeffs,bw_high[hlfpnt1_high]]), 1.0e-13)
    print("Memory usage 3 is",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024)
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

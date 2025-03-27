import os, random, time, resource
import numpy as np
from scipy.linalg import lstsq
import pandas as pd

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

energy_selector = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/energy_selector.npy")
force_selector = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/force_selector.npy")
aw = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/aw.npy")
bw_high = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/bw_high.npy")
df = pd.read_csv("df_diag.csv",index_col=0)
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
hlfpnt1 = np.array([item for sublist_index in ind_configs_index[:hlfpnt] for item in configs_index[sublist_index]])
hlfpnt1_en = hlfpnt1[np.where(energy_selector[hlfpnt1])]
aw_en = aw[hlfpnt1_en]
hlfpnt2 = [item for sublist_index in ind_configs_index[hlfpnt:] for item in configs_index[sublist_index]]

probabilities = df["e_diag"].values/df["e_diag"].sum()
slctd_num_init = 50
slctd = np.random.choice(df.index,slctd_num_init,replace=False,p=probabilities)
unslctd = np.setdiff1d(df.index,slctd)
hlfpnt1_slctd = np.array([item for slctd_ind in slctd for item in configs_index[ind_configs_index[slctd_ind]]])
hlfpnt1_unslctd = np.array([item for unslctd_ind in unslctd for item in configs_index[ind_configs_index[unslctd_ind]]])

u, s, vh = np.linalg.svd(aw[hlfpnt1_slctd], full_matrices=False)
XTX = aw[hlfpnt1_slctd].T @ aw[hlfpnt1_slctd]
# XTX_inv = np.linalg.inv(XTX)
XTX_inv = vh.T @ np.diag(np.reciprocal(s)**2) @ vh
# XTy_low = aw[hlfpnt1].T @ bw_low[hlfpnt1].reshape(-1,1)

for i in range(2952):

    print(i)
    
    if slctd.shape[0] in [50,100,200,350,500,750,1000,1500,2250,3000]:
        print("\nFitting to selected "+str(slctd.shape[0])+" high precision configurations only")
        coefficients, *_ = lstsq(aw[hlfpnt1_slctd], bw_high[hlfpnt1_slctd], 1.0e-13)
        train_residual = np.square(np.dot(aw[hlfpnt1],coefficients) - bw_high[hlfpnt1])
        print("Energy training RMSE is", np.sqrt(np.sum(train_residual*energy_selector[hlfpnt1]/22500)/energy_selector[hlfpnt1].sum()))
        print("Force training RMSE is", np.sqrt(np.sum(train_residual*force_selector[hlfpnt1])/force_selector[hlfpnt1].sum()))
        train_residual = np.square(np.dot(aw[hlfpnt1_slctd],coefficients) - bw_high[hlfpnt1_slctd])
        print("Energy training selected high precision points RMSE is", np.sqrt(np.sum(train_residual*energy_selector[hlfpnt1_slctd]/22500)/energy_selector[hlfpnt1_slctd].sum()))
        print("Force training selected high precision points RMSE is", np.sqrt(np.sum(train_residual*force_selector[hlfpnt1_slctd])/force_selector[hlfpnt1_slctd].sum()))
        test_residual = np.square(np.dot(aw[hlfpnt2],coefficients) - bw_high[hlfpnt2])
        print("Energy testing RMSE is", np.sqrt(np.sum(test_residual*energy_selector[hlfpnt2]/22500)/energy_selector[hlfpnt2].sum()))
        print("Force testing RMSE is", np.sqrt(np.sum(test_residual*force_selector[hlfpnt2])/force_selector[hlfpnt2].sum()))
        entire_test_diff = np.dot(aw,coefficients) - bw_high
        entire_test_residual = np.square(entire_test_diff)
        print("Energy RMSE if tested on the entire dataset is", np.sqrt(np.sum(entire_test_residual*energy_selector/22500)/energy_selector.sum()))
        print("Force RMSE if tested on the entire dataset is", np.sqrt(np.sum(entire_test_residual*force_selector)/force_selector.sum()))

    # Get coefficients and residuals
    coeffs = XTX_inv @ (aw[hlfpnt1_slctd].T @ bw_high[hlfpnt1_slctd].reshape(-1,1))
    en_residuals_sq = np.square(aw_en @ coeffs - bw_high[hlfpnt1_en].reshape(-1,1)).reshape(-1)
    # print(np.sqrt(np.mean(en_residuals_sq[slctd]))/150,np.sqrt(np.mean(en_residuals_sq[unslctd]))/150,np.sqrt(np.mean(en_residuals_sq))/150)

    # Get extended leverage scores
    fake_lev_scores = aw_en @ XTX_inv
    fake_lev_scores = np.einsum('ij,ji->i', fake_lev_scores, aw_en.T)

    # Get Cook's distance and select highest Cook's distance configuration
    new_e_cooks = en_residuals_sq * fake_lev_scores / (1+fake_lev_scores)
    df["e_cooks"] = new_e_cooks
    index_to_add = df.loc[unslctd,"e_cooks"].idxmax()

    # Update slctd, unslctd and hlfpnt1_slctd
    slctd = np.append(slctd,index_to_add)
    unslctd = np.setdiff1d(unslctd,index_to_add)
    hlfpnt1_slctd = np.append(hlfpnt1_slctd,configs_index[ind_configs_index[index_to_add]])
    hlfpnt1_add = np.array(configs_index[ind_configs_index[index_to_add]])

    # Update XTX_inv
    left_update = XTX_inv @ aw[hlfpnt1_add].T
    inv_update = np.linalg.inv(np.eye(len(hlfpnt1_add)) + aw[hlfpnt1_add] @ left_update)
    right_update = aw[hlfpnt1_add] @ XTX_inv
    XTX_inv -= left_update @ inv_update @ right_update
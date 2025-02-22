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

aw = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/aw.npy")
bw_low = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/bw_lower.npy")
bw_high = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/bw_high.npy")
energy_selector = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/energy_selector.npy")
force_selector = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/force_selector.npy")
# coeffs = np.load("/vast/home/baghishov/old_entropy/qSNAP/fidelity/Be/lowerBe_reg/coefficients.npy")
df = pd.read_csv("df_cooks.csv",index_col=0)
df_diag = pd.read_csv("df_diag.csv",index_col=0)
# u = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/svd/Be/u.npy")
s = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/svd/Be/s.npy")
vh = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/svd/Be/vh.npy")
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
XTX = aw[hlfpnt1].T @ aw[hlfpnt1]
# XTX_inv = np.linalg.inv(XTX)
XTX_inv = vh.T @ np.diag(np.reciprocal(s)**2) @ vh
XTy_low = aw[hlfpnt1].T @ bw_low[hlfpnt1].reshape(-1,1)

for i in range(hlfpnt-10):

    print(i)

    if i != 0:
        index_to_remove = df['e_cooks'].idxmin()
        # print("Removing",index_to_remove)
        # print(df.loc[index_to_remove,:])
        df = df.drop(index_to_remove)
        df_diag = df_diag.drop(index_to_remove)
        hlfpnt1_rmvd = np.array(configs_index[ind_configs_index[index_to_remove]])
    slctd = df.index
    hlfpnt1_high = np.array([item for slctd_ind in slctd for item in configs_index[ind_configs_index[slctd_ind]]])
    hlfpnt1_high_en = hlfpnt1_high[np.where(energy_selector[hlfpnt1_high])]
    aw_slctd_en = aw[hlfpnt1_high_en]

    if slctd.shape[0] in [100,200,350,500,750,1000,1500,2250,3000]:
        print("\nFitting to selected "+str(slctd.shape[0])+" high precision configurations only")
        coefficients, *_ = lstsq(aw[hlfpnt1_high], bw_high[hlfpnt1_high], 1.0e-13)
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

    if i!=0:
        left_update = XTX_inv @ aw[hlfpnt1_rmvd].T
        inv_update = np.linalg.inv(np.eye(len(hlfpnt1_rmvd)) - aw[hlfpnt1_rmvd] @ left_update)
        right_update = aw[hlfpnt1_rmvd] @ XTX_inv
        XTX_inv += left_update @ inv_update @ right_update
        XTy_low -= aw[hlfpnt1_rmvd].T @ bw_low[hlfpnt1_rmvd].reshape(-1,1)
    coeffs_low = XTX_inv @ XTy_low
    en_residuals_squared_low = np.square(aw_slctd_en @ coeffs_low - bw_low[hlfpnt1_high_en].reshape(-1,1)).reshape(-1)
    # print("rmse with train low is",1000/150*np.sqrt(np.mean(en_residuals_squared_low)))
    # print("rmse with train high is",1000/150*np.sqrt(np.mean(np.square(aw_slctd_en @ coeffs_low - bw_high[hlfpnt1_high_en].reshape(-1,1)).reshape(-1))))
    leverage_score = aw_slctd_en @ XTX_inv
    leverage_score = np.einsum('ij,ji->i',leverage_score,aw_slctd_en.T)
    # print(leverage_score.shape)
    # print(df_diag.shape)
    # for j in range(df_diag.shape[0]):
    #     print(df_diag.iloc[j,0],leverage_score[j])
    new_e_cooks = en_residuals_squared_low * leverage_score / (1-leverage_score)**2
    # print(new_e_cooks.shape)
    # print(df.shape)
    # for j in range(df.shape[0]):
    #     print(df.iloc[j,0],new_e_cooks[j])
    df['e_cooks'] = new_e_cooks
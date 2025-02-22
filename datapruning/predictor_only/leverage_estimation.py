import os, random, time
from ase.geometry import get_distances
from copy import deepcopy
import numpy as np
import pandas as pd
# from mpi4py import MPI
from scipy.linalg import lstsq
from fitsnap3lib.fitsnap import FitSnap
from fitsnap3lib.scrapers.ase_funcs import get_apre, ase_scraper
import pandas
from sys import float_info as fi
# os.chdir("lanl/W/")

def ase_scraper(snap, frames, energies, forces, stresses):
    """
    Function to organize groups and allocate shared arrays used in Calculator. For now when using 
    ASE frames, we don't have groups.

    Args:
        s: fitsnap instance.
        data: List of ASE frames or dictionary group table containing frames.

    Returns a list of data dictionaries suitable for fitsnap descriptor calculator.
    If running in parallel, this list will be distributed over procs, so that each proc will have a 
    portion of the list.
    """

    snap.data = [collate_data(snap, indx, len(frames), a, e, f, s) for indx, (a,e,f,s) in enumerate(zip(frames, energies, forces, stresses))]
    # Simply collate data from Atoms objects if we have a list of Atoms objecst.
    # if type(frames) == list:
        # s.data = [collate_data(atoms) for atoms in data]
    # If we have a dictionary, assume we are dealing with groups.
    # elif type(data) == dict:
    #     assign_validation(data)
    #     snap.data = []
    #     for name in data:
    #         frames = data[name]["frames"]
    #         # Extend the fitsnap data list with this group.
    #         snap.data.extend([collate_data(atoms, name, data[name]) for atoms in frames])
    # else:
    #     raise Exception("Argument must be list or dictionary for ASE scraper.")

def collate_data(s, indx, size, atoms, energy, forces, stresses):
    """
    Function to organize fitting data for FitSNAP from ASE atoms objects.

    Args: 
        atoms: ASE atoms object for a single configuration of atoms.
        name: Optional name of this configuration.
        group_dict: Optional dictionary containing group information.

    Returns a data dictionary for a single configuration.
    """

    # Transform ASE cell to be appropriate for LAMMPS.
    apre = get_apre(cell=atoms.cell)
    R = np.dot(np.linalg.inv(atoms.cell), apre)
    positions = np.matmul(atoms.get_positions(), R)
    cell = apre.T

    # Make a data dictionary for this config.

    data = {}
    data['PositionsStyle'] = 'angstrom'
    data['AtomTypeStyle'] = 'chemicalsymbol'
    data['StressStyle'] = 'bar'
    data['LatticeStyle'] = 'angstrom'
    data['EnergyStyle'] = 'electronvolt'
    data['ForcesStyle'] = 'electronvoltperangstrom'
    data['Group'] = 'All'
    data['File'] = None
    data['Stress'] = stresses
    data['Positions'] = positions
    data['Energy'] = energy
    data['AtomTypes'] = atoms.get_chemical_symbols()
    data['NumAtoms'] = len(atoms)
    data['Forces'] = forces
    data['QMLattice'] = cell
    data['test_bool'] = indx>=s.config.sections["GROUPS"].group_table["All"]["training_size"]*size
    data['Lattice'] = cell
    data['Rotation'] = np.array([[1,0,0],[0,1,0],[0,0,1]])
    data['Translation'] = np.zeros((len(atoms), 3))
    data['eweight'] = s.config.sections["GROUPS"].group_table["All"]["eweight"]
    data['fweight'] = s.config.sections["GROUPS"].group_table["All"]["fweight"]
    data['vweight'] = s.config.sections["GROUPS"].group_table["All"]["vweight"]

    return data

def load_files(file_name_structures, file_name_energies):
    df_structures = pandas.read_hdf(file_name_structures)
    df_structures.sort_index(inplace=True)

    df_energies = pandas.read_hdf(file_name_energies)
    df_energies.sort_values(by=["index"], inplace=True)

    df_structures = df_structures[df_structures.index.isin(df_energies["index"].values)]
    return df_structures, df_energies

file_name_structures = "/vast/home/baghishov/old_entropy/lanl_data/Be_large_subset_structures.h5"
file_name_energies = "/vast/home/baghishov/old_entropy/lanl_data/Be_high_3.hdf"
atoms_clmn = "ASEatoms_rescale"
df_structures, df_energies = load_files(file_name_structures, file_name_energies)
configs_num = df_structures[atoms_clmn].shape[0]

del df_energies

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

aw = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/aw.npy")[hlfpnt1]
coeffs_low = np.load("/vast/home/baghishov/old_entropy/qSNAP/fidelity/Be/lowBe_reg/coefficients.npy")
y_low = aw @ coeffs_low
print(y_low.size == len(hlfpnt1))
# del aw
bw_high = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/bw_high.npy")
bw_low = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/bw_low.npy")
bw_diff = bw_high - bw_low
energy_selector = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/energy_selector.npy")
force_selector = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/npy_files/Be/force_selector.npy")

start_time = time.time()
u = np.load("/vast/home/baghishov/old_entropy/qSNAP/big_files/svd/Be/u.npy")
print(u.shape[0] == len(hlfpnt1))
# i_strt = 0
# delta_y = np.zeros((len(hlfpnt1),hlfpnt))
XTX_diag = (aw.T @ aw).diagonal()

df = pd.DataFrame(np.zeros((hlfpnt,6)),columns=['e_diag','f_diag_rmse','f_diag_mae','f_diag_sum','det_cov_mat','det_cov_mat_reg'])
diag_elms = np.sum(u**2,axis=1)
np.save("diag_elms.npy", diag_elms)
i_strt = 0
for i,j in enumerate(ind_configs_index[:hlfpnt]):
    n_atms = len(df_structures[atoms_clmn].values[j])
    # s = np.linalg.svd(aw[(i_strt):(i_strt+1+3*n_atms)],compute_uv=False)
    temp_mat = aw[(i_strt):(i_strt+1+3*n_atms)] - aw.mean(axis=0)
    # print(temp_mat.shape)
    s = np.linalg.svd(temp_mat,full_matrices=False)[1]/1e4
    df.loc[i,'e_diag'] = diag_elms[i_strt]
    df.loc[i,'f_diag_rmse'] = np.sqrt(np.mean(np.square(diag_elms[(i_strt+1):(i_strt+1+3*n_atms)])))
    df.loc[i,'f_diag_mae'] = np.mean(np.abs(diag_elms[(i_strt+1):(i_strt+1+3*n_atms)]))
    df.loc[i,'f_diag_sum'] = np.sum(diag_elms[(i_strt+1):(i_strt+1+3*n_atms)])
    df.loc[i,'det_cov_mat'] = np.prod(s[s>1e-17])
    df.loc[i,'det_cov_mat_reg'] = np.prod(np.linalg.det(temp_mat.T @ temp_mat + 0.008*np.eye(temp_mat.shape[1])))
    i_strt += 1+3*n_atms
    print(i, df.loc[i,'det_cov_mat'], df.loc[i,'det_cov_mat_reg'])

print(i_strt)
df.to_csv('df_diag.csv')

# df = pd.DataFrame(np.zeros((hlfpnt,4)),columns=['e_rmse','f_rmse','e_rmse_wo','f_rmse_wo'])
# for i,j in enumerate(ind_configs_index[:hlfpnt]):
#     hlfpnt1_high = [item for item in configs_index[j]]
#     hlfpnt1_low = [item for item in hlfpnt1 if item not in hlfpnt1_high]
#     time0 = time.time()
#     delta_y[:,i] = u @ u[i_strt:(i_strt+len(hlfpnt1_high))].T @ bw_diff[hlfpnt1_high]
#     errors = np.square(y_low + delta_y[:,i] - bw_high[hlfpnt1])
#     df.loc[i,'e_rmse'] = np.sqrt(np.sum(errors*energy_selector[hlfpnt1]/22500)/energy_selector[hlfpnt1].sum())
#     df.loc[i,'f_rmse'] = np.sqrt(np.sum(errors*force_selector[hlfpnt1])/force_selector[hlfpnt1].sum())
#     energy_selector[hlfpnt1_high][0] = 0
#     force_selector[hlfpnt1_high][1:] = 0
#     df.loc[i,'e_rmse_wo'] = np.sqrt(np.sum(errors*energy_selector[hlfpnt1]/22500)/energy_selector[hlfpnt1].sum())
#     df.loc[i,'f_rmse_wo'] = np.sqrt(np.sum(errors*force_selector[hlfpnt1])/force_selector[hlfpnt1].sum())
#     energy_selector[hlfpnt1_high][0] = 1
#     force_selector[hlfpnt1_high][1:] = 1
#     i_strt += len(hlfpnt1_high)
#     print(i,j,1000*df.loc[i,"e_rmse"],time.time()-time0)

# np.save("delta_y.npy", delta_y)
# df.to_csv('df.csv')
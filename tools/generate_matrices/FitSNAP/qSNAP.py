import random, time, resource
import numpy as np
from mpi4py import MPI
from scipy.linalg import lstsq
from fitsnap3lib.fitsnap import FitSnap
from fitsnap3lib.scrapers.ase_funcs import get_apre, ase_scraper
import pandas


def ase_scraper(snap, frames, energies, forces, stresses):
    snap.data = [collate_data(snap, indx, len(frames), a, e, f, s) for indx, (a,e,f,s) in enumerate(zip(frames, energies, forces, stresses))]

def collate_data(s, indx, size, atoms, energy, forces, stresses):

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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ACE_settings = \
{
"BISPECTRUM":
    {
    "numTypes": 1,
    "twojmax": 6,
    "rcutfac": 4.812302818,
    "rfac0": 0.99363,
    "rmin0": 0.0,
    "wj": 1.0,
    "radelem": 0.5,
    "type": "Be",
    "wselfallflag": 0,
    "chemflag": 0,
    "bzeroflag": 1,
    "quadraticflag": 0,
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1,
    "force": 1,
    "stress" : 0,
    },
"ESHIFT":
    {
    "Be" : 0.0
    },
"GROUPS":
    {
    # name size eweight fweight vweight
    "group_sections" : "name training_size testing_size eweight fweight vweight",
    "group_types" : "str float float float float float",
    "smartweights" : 0,
    "random_sampling" : 0,
    "All" :  "0.5    0.5    150.0      1.0  0.0"
    },
"OUTFILE":
    {
    "metrics" : "Be_metrics.md",
    "potential" : "Be_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "zero 10.0",
    "pair_coeff": "* *",
    },
"SOLVER":
    {
    "solver": "SVD",
    "compute_testerrs": 1,
    "detailed_errors": 1
    },
"EXTRAS":
    {
    "dump_descriptors": 0,
    "dump_truth": 0,
    "dump_weights": 0,
    "dump_dataframe": 0
    },
"MEMORY":
    {
    "override": 0
    }
}
fs_instance1 = FitSnap(ACE_settings, comm=comm, arglist=["--overwrite"])
file_name_structures = "Be_large_subset_structures.h5"
file_name_energies = "Be_high.hdf"
atoms_clmn = "ASEatoms_rescale"
df_structures, df_energies = load_files(file_name_structures, file_name_energies)
configs_num = df_structures[atoms_clmn].shape[0]
ratio = configs_num//size
rem = configs_num%size
a1 = rank*ratio + min(rank,rem)
a2 = (rank+1)*ratio + min(rank,rem-1) + 1
ase_scraper(fs_instance1, df_structures[atoms_clmn].values[a1:a2], df_energies['energy'].values[a1:a2], df_energies["forces"].values[a1:a2], df_energies["stress"].values[a1:a2])
fs_instance1.process_configs(allgather=True)

del df_energies

if rank == 0:

    config_idxs = []
    for i in range(configs_num):
        config_idxs.append([i for _ in range(1+3*len(df_structures[atoms_clmn].values[i]))])

    w = fs_instance1.pt.shared_arrays["w"].array
    a = fs_instance1.pt.shared_arrays["a"].array
    b = fs_instance1.pt.shared_arrays["b"].array
    # energy_selector = np.where(fs_instance1.pt.shared_arrays["w"].array == 150, 1, 0)

    np.save('X.npy',a)
    np.save('y.npy',b)
    np.save('w.npy',w)
    np.save('config_idxs.npy',np.concatenate(config_idxs))
    # np.save('enrow_mask.npy',energy_selector.astype(bool))

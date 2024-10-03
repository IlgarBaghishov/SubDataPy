#!/usr/bin/env python
import os, sys, torch
import numpy as np
import time
import pickle
from pyamff.utilities.logTool import setLogger, writeSysInfo
from pyamff.neighborlist import NeighborLists
from pyamff.utilities.preprocessor import fetchProp
from pyamff.utilities.preprocessor import Scaler
from pyamff.config import ConfigClass
from pyamff.fingerprints.fingerprints import Fingerprints
from ase.io import Trajectory
torch.set_printoptions(threshold=100000000)
np.set_printoptions(threshold=100000000)
#Read and set up setting parameters
print(os.getcwd())
config = ConfigClass()
config.initialize()
#logger = setLogger()
#writeSysInfo(logger)

#Fetch fingerprint parameters in Format: {'H':[G1, G2], 'Pd':[G1, G2]}
fp_paras = config.config['fp_paras'].fp_paras

#Read in images
images = Trajectory(config.config['trajectory_file'], 'r')

#Convert fingerprint papamter objects and store in a list
nFPs = {}
for key in fp_paras.keys():
   nFPs[key] = len(fp_paras[key])

#Preprocess and check the properties and images
scaler = Scaler(scalerType=config.config['scaler_type'])
scaler = Scaler.set_scaler(scaler)
trainingimages, properties, scaler = fetchProp(images, scaler=scaler, forceTraining=True)
srcData = list(trainingimages.keys())

# Do the calculation
fpcalc = Fingerprints(uniq_elements=config.config['fp_paras'].uniq_elements, filename = config.config['fp_parameter_file'], nfps = nFPs)
trainingimages_len = len(trainingimages)
struct_row = np.zeros(trainingimages_len+1, dtype=int)
all_fps_len = 0
for i in range(trainingimages_len):
    all_fps_len += trainingimages[i].positions.shape[0]
    struct_row[i+1] = all_fps_len

max_N_fps = fpcalc.max_nfps
min_fps = 10000*np.ones(max_N_fps)
max_fps = -10000*np.ones(max_N_fps)
# all_fps = []
t1 = time.time()
# for i, struct in enumerate(trainingimages):
#     if struct < 0:
#         continue
# #    logger.info('  Calculating FPs for image %d', struct)
#     else:
#         print('Calculating FPs for image',struct)
#         chemsymbols = trainingimages[struct].get_chemical_symbols()
#         fingerprints, fingerprintprimes = fpcalc.calcFPs(trainingimages[struct], chemsymbols)
#         # print(fingerprints)
#         # print(all_fps)
#         all_fps.append(fingerprints)
#         min_fps = np.vstack([fingerprints,min_fps]).min(axis=0)
#         max_fps = np.vstack([fingerprints,max_fps]).max(axis=0)
#         # print("size of all_fps is ",sys.getsizeof(all_fps))
# t2 = time.time()
# print("time of calculating fingerprints is ", t2-t1, "seconds")
# with open('all_fps.pickle','wb') as all_fps_pickle:
#     pickle.dump(all_fps,all_fps_pickle)
# with open('fps_minmax.pickle','wb') as fps_minmax_pickle:
#     pickle.dump([min_fps,max_fps],fps_minmax_pickle)

with open('all_fps.pickle','rb') as all_fps_pickle:
    all_fps2 = pickle.load(all_fps_pickle)
with open('fps_minmax.pickle','rb') as fps_minmax_pickle:
    min_fps,max_fps = pickle.load(fps_minmax_pickle)
print("Number of fingerprints is", len(all_fps2))
t2 = time.time()

all_fps2 = [all_fps2[i]/max_fps for i in range(0,len(all_fps2))]  # normalizes all fingerprints

# N_grids_fps = 2*np.ones(max_N_fps)
# min_fps_ = min_fps - (max_fps - min_fps)/N_grids_fps/2
# max_fps_ = max_fps + (max_fps - min_fps)/N_grids_fps/2
# d_fps = (max_fps_ - min_fps_) / N_grids_fps
for d_fps in np.arange(0.065,0.07,0.05):  # 0.1is1079 0.2is199
    print("\nTrying delta G of ", d_fps)
    all_fps = all_fps2.copy()
    d_fps2 = np.square(d_fps)
    # N_digits_fps = [len(str(i)) for i in N_grids_fps-1]

    removed = []
    # hash_table = {}
    unique_bank = np.zeros((all_fps_len//8+1,max_N_fps))
    start_row = 0
    end_row = struct_row[-1]-struct_row[-2]
    unique_bank[start_row:end_row,:] = all_fps[-1]
    all_fps.pop()
    for i in range(trainingimages_len-2,-1,-1):  # loop through structures in all_fps list
        print("Structure",i, 20051-i-len(removed))
        unique = False
        curr_struct = all_fps[i]
        for j in range(all_fps[i].shape[0]):  # loop through atoms in a structure
            all_fps_subset = unique_bank[:end_row]
            ll = all_fps_subset > (curr_struct[j]-d_fps)
            hh = all_fps_subset < (curr_struct[j]+d_fps)
            all_fps_subset = all_fps_subset[np.all(ll & hh, axis=1)]
            # print(j)
            # print(all_fps_subset.shape[0])
            # print("Atom ",j-struct_row[i]," - ",all_fps_subset.shape[0]/j*100)
            if all_fps_subset.shape[0] == 0:
                unique = True
                break
            # for k in range(all_fps_subset.shape[0]):
            if ~np.any(np.sum(np.square(all_fps_subset - curr_struct[j]),axis=1)/max_N_fps < d_fps2):
                unique = True
                break
        if unique == False:
            removed.append(i)
        else:
            start_row = end_row
            end_row = start_row + struct_row[i+1]-struct_row[i]
            try:
                unique_bank[start_row:end_row,:] = all_fps[i]
            except:
                unique_bank = np.append(unique_bank,np.zeros((unique_bank.shape[0],max_N_fps)),axis=0)
                unique_bank[start_row:end_row,:] = all_fps[i]
        all_fps.pop()

    # print(removed)
    chosen = [i for i in srcData if i not in removed]
    print(chosen)
    removed_len = len(removed)
    print("Removed",removed_len)
    chosen_len = trainingimages_len-len(removed)
    print("Chosen",chosen_len)
    t3 = time.time()
    print("time of finding unique fingerprints is ", t3-t2, "seconds")
    print("total elapsed time is", t3-t1, "seconds")

    if chosen_len > 0.9*trainingimages_len or chosen_len < 0.02*trainingimages_len:
        continue

    print("creating directory and files including unique traj file...")
    output_traj_directory = "../%3.2f_%d/" % (d_fps, chosen_len)
    if not os.path.isdir(output_traj_directory):
        os.mkdir(output_traj_directory)
    os.chdir(output_traj_directory)
    output_traj = Trajectory(config.config['trajectory_file'],'w')
    for i in range(trainingimages_len):
        if i not in removed:
            output_traj.write(images[i])
    output_traj.close()
    # os.system("cp ../entire_traj/config.ini .") # remember this config has restart=True
    # os.system("cp ../entire_traj/run.sh .")
    # os.system("cp ../entire_traj/pyamff.pt .")
    # os.system("cp ../entire_traj/fpParas.dat .")
    # with open("run.sh","r") as run_r:
    #     lines = run_r.readlines()
    #     with open("run.sh","w") as run_w:
    #         for line in lines:
    #             if "#SBATCH -J" in line:
    #                 run_w.write("#SBATCH -J " + output_traj_directory[3:-1] + "\n")
    #             else:
    #                 run_w.write(line)
    # os.system("cp ../entire_traj/test.py .")
    # os.system("cp ../entire_traj/run_test.sh .")
    # with open("run_test.sh","r") as run_r:
    #     lines = run_r.readlines()
    #     with open("run_test.sh","w") as run_w:
    #         for line in lines:
    #             if "#SBATCH -J" in line:
    #                 run_w.write("#SBATCH -J test_" + output_traj_directory[3:-1] + "\n")
    #             else:
    #                 run_w.write(line)
    # os.system("sbatch run.sh")

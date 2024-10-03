#!/usr/bin/env python
import os, sys, torch
import numpy as np
from scipy.special import xlogy
import time
import pickle
from random import sample, random, randint
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
image_size = np.zeros(trainingimages_len, dtype=int)
for i in range(trainingimages_len):
    image_size[i] = trainingimages[i].positions.shape[0]

max_N_fps = fpcalc.max_nfps
# min_fps = 10000*np.ones(max_N_fps)
# max_fps = -10000*np.ones(max_N_fps)
# all_fps = []
# t1 = time.time()
# for i, struct in enumerate(trainingimages):
#     if struct < 0:
#         continue
# #    logger.info('  Calculating FPs for image %d', struct)
#     else:
#         print('Calculating FPs for image',struct)
#         chemsymbols = trainingimages[struct].get_chemical_symbols()
#         fingerprints, fingerprintprimes = fpcalc.calcFPs(trainingimages[struct], chemsymbols)
#         all_fps.append(fingerprints)
#         min_fps = np.vstack([fingerprints,min_fps]).min(axis=0)
#         max_fps = np.vstack([fingerprints,max_fps]).max(axis=0)
# t2 = time.time()
# print("time of calculating fingerprints is ", t2-t1, "seconds")
# with open('all_fps.pickle','wb') as all_fps_pickle:
#     pickle.dump(all_fps,all_fps_pickle)
# with open('fps_minmax.pickle','wb') as fps_minmax_pickle:
#     pickle.dump([min_fps,max_fps],fps_minmax_pickle)

with open('all_fps.pickle','rb') as all_fps_pickle:
    all_fps = pickle.load(all_fps_pickle)
with open('fps_minmax.pickle','rb') as fps_minmax_pickle:
    min_fps,max_fps = pickle.load(fps_minmax_pickle)
t2 = time.time()

bin_size = 10
# bin_edges = np.linspace(min_fps,max_fps,bin_size+1)
fps_bins = np.zeros((trainingimages_len,max_N_fps,bin_size))
for i in range(trainingimages_len):
    for j in range(max_N_fps):
        fps_bins[i,j] = np.histogram(all_fps[i][:,j], bins=bin_size, range=(min_fps[j],max_fps[j]))[0]
t3 = time.time()
print("time of calculating fps_bins is ", t3-t2, "seconds")

stepsize = 150000
# percent_of_data = 0.2
# chosen_len = int(percent_of_data*trainingimages_len)
exp_factor = 150
chosen_len = 2250
# for chosen_len in range(500,2500,250):
chosen = sample(srcData,chosen_len)
# chosen_list = trainingimages[chosen]
removed = [i for i in srcData if i not in chosen]
print("Initial calculation of entropy with", chosen_len, "random data points")
print("Chose",chosen)
max_entropy = max_N_fps*np.log(bin_size)

chosen_atoms_numb = np.sum(image_size[chosen])  # calculates total number of atoms in all structures in chosen dataset
fps_dist = np.sum(fps_bins[chosen],axis=0)/chosen_atoms_numb  # calculates distribution of fingerprints by summing number of atoms in each bin in all structures and then dividing it by total number of atoms in all structures in chosen dataset
fps_dist_init = fps_dist.copy()
entropy_per_fp_old = np.sum(xlogy(fps_dist,fps_dist),axis=1)  # calculates minus entropy for each fingerprint
entropy_old = np.sum(entropy_per_fp_old)/max_entropy  # add minus entropies of each fingerprint to get total minus entropy that needs to be minimized
print("%15.10f" % entropy_old)
t4 = time.time()
print("time of calculating entropy is ", t4-t3, "seconds")

step_entropy = np.zeros((stepsize,2))
step_entropy[0] = [0,entropy_old]
for step in range(1,stepsize):
    print("\nStep ", step)
    
    add = sample(removed,1)[0]
    rem = sample(chosen,1)[0]
    removed.remove(add)
    removed.append(rem)
    chosen.remove(rem)
    chosen.append(add)
    removed_len = len(removed)
    chosen_len = len(chosen)

    chosen_atoms_numb = np.sum(image_size[chosen])  # calculates total number of atoms in all structures in chosen dataset
    fps_dist = np.sum(fps_bins[chosen],axis=0)/chosen_atoms_numb  # calculates distribution of fingerprints by summing number of atoms in each bin in all structures and then dividing it by total number of atoms in all structures in chosen dataset
    entropy_per_fp = np.sum(xlogy(fps_dist,fps_dist),axis=1)  # calculates minus entropy for each fingerprint
    entropy = np.sum(entropy_per_fp)/max_entropy  # add minus entropies of each fingerprint to get total minus entropy that needs to be minimized
    print("Chose",chosen)
    print("%15.10f" % entropy)
    
    if entropy > entropy_old:
        rand = random()
        if rand > np.exp(-exp_factor*chosen_len*(entropy-entropy_old)):  # BE CAREFUL, NEED TO PLAY WITH 500, I AM PRETTY SURE THERE IS A BETTER VALUE
            removed.remove(rem)
            removed.append(add)
            chosen.remove(add)
            chosen.append(rem)
            print("didn't accept this swap")
            continue
    
    entropy_old = entropy
    step_entropy[step] = [step,entropy]
step_entropy = np.vstack([step_entropy[0],step_entropy[np.all(step_entropy, axis=1)]])
print(step_entropy)
t5 = time.time()
print("total elapsed time after loading pickle files is ", t5-t2, "seconds")
print("first step\n",step_entropy[0])
print("last step\n",step_entropy[-1])

print("creating directory and files including unique traj file...")
output_traj_directory = "../%d_%d/" % (exp_factor, chosen_len)
if not os.path.isdir(output_traj_directory):
    os.mkdir(output_traj_directory)
os.chdir(output_traj_directory)
np.savetxt("step_entropy.dat", step_entropy, fmt=['%10d','%12.5f'])
np.savetxt("fps_dist.dat", fps_dist, fmt=['%9.5f','%9.5f','%9.5f','%9.5f','%9.5f','%9.5f','%9.5f','%9.5f','%9.5f','%9.5f'])
np.savetxt("fps_dist_init.dat", fps_dist_init, fmt=['%9.5f','%9.5f','%9.5f','%9.5f','%9.5f','%9.5f','%9.5f','%9.5f','%9.5f','%9.5f'])
os.system("cp ../main/postprocess.ipynb .")
output_traj = Trajectory(config.config['trajectory_file'],'w')
for i in range(trainingimages_len):
    if i in chosen:
        output_traj.write(images[i])
output_traj.close()
os.system("cp ../entire_traj/config.ini .") # remember this config has restart=True
with open("config.ini","r") as config_r:
    lines = config_r.readlines()
    with open("config.ini","w") as config_w:
        for line in lines:
            if "batch_num_per_proc" in line:
                batch_num_per_proc = round(chosen_len/150/config.config['process_num'])
                if batch_num_per_proc == 0:
                    config_w.write("batch_num_per_proc = 1\n")
                elif batch_num_per_proc > 0:
                    config_w.write("batch_num_per_proc = " + str(batch_num_per_proc) + "\n")
            elif "master_port" in line:
                config_w.write("master_port = " + str(randint(1,5)) + str(randint(1,5)) + str(randint(1,5)) + str(randint(1,5)) + str(randint(1,5)) + "\n")
            else:
                config_w.write(line)
os.system("cp ../entire_traj/run.sh .")
os.system("cp ../entire_traj/pyamff.pt .")
os.system("cp ../entire_traj/fpParas.dat .")
with open("run.sh","r") as run_r:
    lines = run_r.readlines()
    with open("run.sh","w") as run_w:
        for line in lines:
            if "#SBATCH -J" in line:
                run_w.write("#SBATCH -J " + output_traj_directory[3:-1] + "\n")
            else:
                run_w.write(line)
os.system("cp ../entire_traj/test.py .")
os.system("cp ../entire_traj/run_test.sh .")
with open("run_test.sh","r") as run_r:
    lines = run_r.readlines()
    with open("run_test.sh","w") as run_w:
        for line in lines:
            if "#SBATCH -J" in line:
                run_w.write("#SBATCH -J test_" + output_traj_directory[3:-1] + "\n")
            else:
                run_w.write(line)
os.system("sbatch run.sh")

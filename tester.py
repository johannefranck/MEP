import matplotlib.pyplot as plt
import numpy as np
import data
import plot_functions
from pymatreader import read_mat

main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
filelist = data.get_all_paths(main_path)
'''
#remove the files with no signal, or little signal
removes = ["/X96343_coil_orient.mat", 
           "/X02035_coil_orient.mat",
           "/sub-X76928_ses-1_task-coilorientation_emg.mat",
           "/sub-X98504_ses-1_task-coilorientation_emg.mat"]
'''

print(filelist)

'''
#print fulllength eksempler
path = main_path + "/sub-X99909_ses-1_task-coilorientation_emg.mat"
data1 = read_mat(path)
if "sub" in path:
    key = list(data1.keys())[3]
else:
    key = list(data1.keys())[0]

X_raw = data1[key]['values'][:,0]
y = data1[key]['frameinfo']['state']
X_raw, y = data.delete_frames(X_raw,y)
print("hey")
plt.plot(X_raw)
plt.show()

'''

#plot different pr subject plots
#plot_functions.plot_groups(X, groups, specifics = [5,18,19,31,32], list_subjects=list_subjects)
#plot_functions.plot_coil(X,y,groups, mean = False, subject = None)

#subject = 21 # set specific subject
#plot_functions.plot_subject_coil(X,y,groups,mean=False,subject=subject)

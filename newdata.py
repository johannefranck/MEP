import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import data
from pymatreader import read_mat

main_path = "/mnt/projects/USS_MEP/NEW_DATA /TMSEEG_CoilOrientation"
data1 = read_mat(main_path + "/analyzed_sub-X06922_ses-000_task-APPA_side-right_emg.mat")
databi1 = read_mat(main_path + "/analyzed_sub-X06922_ses-000_task-Biphasic_APPA_M1_B70_110RMT_APPA_MEPdata_side-right_emg.mat")
databi2 = read_mat(main_path + "/analyzed_sub-X06922_ses-000_task-Biphasic_APPA_M1_B70_95RMT_APPA_MEPdata_side-right_emg.mat")
databi3 = read_mat(main_path + "/analyzed_sub-X06922_ses-000_task-Biphasic_PAAP_M1_B70_110RMT_APPA_MEPdata_side-right_emg.mat")
databi4 = read_mat(main_path + "/analyzed_sub-X06922_ses-000_task-Biphasic_PAAP_M1_B70_110RMT_PAAP_MEPdata_side-right_emg.mat")
data2 = read_mat(main_path + "/analyzed_sub-X06922_ses-001_task-APPA_side-right_emg.mat")
data3 = read_mat(main_path + "/analyzed_sub-X06922_ses-000_task-PAAP_side-right_emg.mat")

#data1 = read_mat(main_path + "/analyzed_sub-X41112_ses-000_task-APPA_side-right_emg.mat")
#data2 = read_mat(main_path + "/analyzed_sub-X41112_ses-001_task-APPA_side-right_emg.mat")
#data3 = read_mat(main_path + "/analyzed_sub-X41112_ses-000_task-PAAP_side-right_emg.mat")


X_raw1 = data1['analyzedData']['values']
y1 = data1['analyzedData']['state']


# plot the lines
plt.plot(np.transpose(data1['analyzedData']['values'])[2025:2110], color = "green")
plt.plot(np.transpose(databi1['analyzedData']['values'])[2025:2110], color = "cyan")
plt.plot(np.transpose(data2['analyzedData']['values'])[2025:2110], color = "blue")
plt.plot(np.transpose(data3['analyzedData']['values'])[2025:2110], color = "red")
plt.plot(np.transpose(databi2['analyzedData']['values'])[2025:2110], color = "pink")
plt.plot(np.transpose(databi3['analyzedData']['values'])[2025:2110], color = "gold")
plt.plot(np.transpose(databi4['analyzedData']['values'])[2025:2110], color = "deepskyblue")

# set the custom legend labels and colors
legend_labels = ["PA: 000_task-APPA_side-right", "biophase1", "biophase2","biophase3","biophase4","PA: 001_task-APPA_side-right", "AP: task-PAAP_side-right"]
legend_colors = ["green", "cyan", "pink", "gold", "deepskyblue", "blue", "red"]
legend_patches = [Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
plt.title("Subject " + str(data1['analyzedData']['sub']))
plt.legend(handles=legend_patches, fontsize=8)
plt.show()


#plt.plot(np.transpose(data1['analyzedData']['values']))
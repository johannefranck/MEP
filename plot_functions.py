import matplotlib.pyplot as plt
import numpy as np
import data 


def plot_groups(X, groups, specifics):
    # Plotting the different subject MEP signals in the same plot, with time on the x axis
    # set the specifics to ['all'] for all subjects otherwise specify: i.e. [3,13,23]

    # Color pre-definitions
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # get a list of colors from the default color cycle in matplotlib
    color_dict = {i: colors[i%len(colors)] for i in range(53)} # create a dictionary with keys from 0 to 51, and values as different colors from the matplotlib library

    # Get the time array with a sampling rate of 2000 Hz
    STAA = 7.5 # sliced_time_after_artifact, here it is 15 timepoints, which is the same a 7.5 ms
    time = np.arange(np.transpose(X).shape[1]) / 2000 * 1000 + STAA

    # Create the figure and axes objects
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    # Plot the data for each group
    lines = []
    if specifics != ["all"]:
        idx = 0
        colors = []
        for group in set(specifics): #{17,7}
            traj_ids = np.where(np.array(groups) == group)[0]
            for i in range(len(traj_ids)):
                line, = axs.plot(time, np.transpose(X)[traj_ids[i], :], c=color_dict[group + idx], alpha=0.5)
                lines.append(line)
            colors.append(color_dict[group + idx])
            idx += 1
            
    else:
        for group in set(groups):
            idx = np.where(np.array(groups) == group)[0]
            for i in range(len(idx)):
                line, = axs.plot(time, np.transpose(X)[idx[i], :], c=color_dict[group], alpha=0.5)
                lines.append(line)
        

    # Set the axis labels and title
    axs.set_xlabel('Time (ms)')
    axs.set_ylabel('mV')
    axs.set_title('MEP signals by subject')

    # Add the legend
    if specifics != ["all"]:
        groups = specifics
    legend_labels = list(set(groups))
    axs.legend(lines, legend_labels, loc='best', prop={'size': 'xx-small'}, title='Subject', title_fontsize='small', framealpha=0.5, facecolor='white', edgecolor='black', labelcolor=colors)

    # Show the plot
    plt.show()



def plot_coil(X,y,groups, mean, subject):
    #plotting coil orientations as PA (blue) and AP (red)
    #specify mean = TRUE if the mean of AP and PA is wanted, otherwise FALSE
    X = np.array(np.transpose(X))
    y = np.array(y)
    groups = np.array(groups)


    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    axs.set_xlabel('Time (ms)')
    axs.set_ylabel('mV')
    if type(subject)==int:
        axs.set_title('MEP signals by Coil orientarion, subject ' + str(list_subjects[subject-1]) + ', groupnr: ' + str(subject) + ', trajs: ' + str(len(y)))
    else:
        axs.set_title('MEP signals by Coil orientarion')

    PA = np.transpose(X[np.where(y==1)])
    AP = np.transpose(X[np.where(y==2)])

    if mean == True:
        axs.set_title('MEP signals by Coil orientarion, Mean')
        plt.plot(np.mean(PA, axis = 1), color = 'blue')
        plt.plot(np.mean(AP, axis = 1), color = 'red')
    else:
        plt.plot(PA, color = 'blue')
        plt.plot(AP, color = 'red')


    axs.legend(['PA','AP'], loc='best', title_fontsize='large', framealpha=0.5, facecolor='white', edgecolor='black', labelcolor=['blue','red'])

    plt.show()


def plot_subject_coil(X,y,groups,list_subjects,mean,subject):
    #plot a subject with PA and AP 
    #mean = True / False, set subject as int for wanted subject
    X = np.array(np.transpose(X))
    y = np.array(y)
    groups = np.array(groups)
    #list_subjects = np.array(list_subjects)

    group_i_where=np.where(groups == subject)[0]
    yi = y[list(group_i_where)]
    Xi = X[list(group_i_where)]
    plot_coil(np.transpose(Xi),yi,groups,mean,subject)



main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"

filelist = data.get_all_paths(main_path)
#filelist = ['/mnt/projects/USS_MEP/COIL_ORIENTATION/X02035_coil_orient.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X05398_coil_orient.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X06188_coil_orient.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X16394_COIL_ORIENT.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X30095_COIL_ORIENT.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X34930_CoilOrientation000.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X36342_COIL_ORIENT.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X37032_coil_orient_000.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X37032_coil_orient_001.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X51231_CoilOrientation.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X58086_COIL_ORIERNT.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X81190_Coil_orientation.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X88230_COIL_ORIENT.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X91583_CoilOrientation000.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X91583_CoilOrientation001.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/X96343_coil_orient.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X01666_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X02299_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X03004_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X13844_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X24208_ses-1_task-coilorientation_run001_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X24208_ses-1_task-coilorientation_run002_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X29554_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X34646_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X36523_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X37945_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X38547_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X40027_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X42448_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X44909_ses-1_task-coilorientation_run001_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X44909_ses-1_task-coilorientation_run002_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X48739_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X55579_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X59029_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X61122_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X64181_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X70786_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X72350_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X74644_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X76928_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X81562_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X85446_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X86768_ses-1_task-coilorientation_run001_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X86768_ses-1_task-coilorientation_run002_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X96010_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X98504_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/sub-X99909_ses-1_task-coilorientation_emg.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/x03172_Coil_orient000.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/x03172_Coil_orient001.mat', '/mnt/projects/USS_MEP/COIL_ORIENTATION/x71487_coil_orient.mat']

X, y, groups, list_subjects = data.get_all_data(filelist)

#plot_groups(X, groups, specifics = [3])
#plot_coil(X,y,groups, mean = False)

subject = 1
plot_subject_coil(X,y,groups,list_subjects,mean=False,subject=subject)
print(filelist)
print(filelist[subject])


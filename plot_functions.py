import matplotlib.pyplot as plt
import numpy as np
import data 
import random


def plot_groups(X, groups, list_subjects, specifics):
    # Plotting the different subject MEP signals in the same plot, with time on the x axis
    # set the specifics to ['all'] for all subjects otherwise specify: i.e. [3,13,23]

    # Color pre-definitions
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # get a list of colors from the default color cycle in matplotlib
    # Create a dictionary with 50 unique color keys
    color_dict = {0: '#D59722', 1: '#ACE7FD', 2: '#D67220', 3: '#194D33', 4: '#48B233', 5: '#9900EF', 6: '#2F9100', 7: '#69D379', 8: '#B6F42F', 9: '#B0659C', 10: '#73830A', 11: '#0D48C2', 12: '#1B5AC3', 13: '#770159', 14: '#68C631', 15: '#877B85', 16: '#4D22B2', 17: '#EC36A1', 18: '#3D7957', 19: '#191D4D', 20: '#A696A3', 21: '#07A66F', 22: '#F4C01E', 23: '#556742', 24: '#2E3A01', 25: '#09AC36', 26: '#4638A2', 27: '#CF1B3B', 28: '#1BCF73', 29: '#3E9AB9', 30: '#EBAF15', 31: '#AB4147', 32: '#4CAF50', 33: '#EDBF24', 34: '#11BBAA', 35: '#F5C1A0', 36: '#42E978', 37: '#2D45B6', 38: '#AF7B0D', 39: '#F7FBD1', 40: '#A8E69B', 41: '#0F174F', 42: '#A62A92', 43: '#523368', 44: '#6E6708', 45: '#0A90CB', 46: '#EAB71B', 47: '#C5C765', 48: '#21329F', 49: '#142BB6'}

    # Get the time array with a sampling rate of 2000 Hz
    STAA = 12.5 # sliced_time_after_artifact, here it is 25 timepoints, which is the same a 12.5 ms
    time = np.arange(np.transpose(X).shape[1]) / 2000 * 1000 + STAA

    # Create the figure and axes objects
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    # Plot the data for each group
    lines = []
    if specifics != ["all"]:

        colors = []
        for group in np.sort(list(set(specifics))): #{17,7}
            traj_ids = np.where(np.array(groups) == group)[0]
            for i in range(len(traj_ids)):
                line, = axs.plot(time, np.transpose(X)[traj_ids[i], :], c=color_dict[group], alpha=0.5)
                lines.append(line)
            colors.append(color_dict[group])
            
    else:
        for group in np.sort(list(set(specifics))):
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
        groups = np.sort(list(specifics))
    
    legend_labels = []
    for i in range(len(groups)):
        legend_labels.append(list_subjects[groups[i]]+str(", ")+str(groups[i]))
    axs.legend(lines, legend_labels, loc='best', prop={'size': 'xx-small'}, title='Subject, groupnr', title_fontsize='small', framealpha=0.5, facecolor='white', edgecolor='black', labelcolor=colors)

    plt.show()



def plot_coil(X,y,list_subjects,groups, mean, subject):
    #plotting coil orientations as PA (blue) and AP (red)
    #specify mean = TRUE if the mean of AP and PA is wanted, otherwise FALSE, subject is set to None
    X = np.array(np.transpose(X))
    y = np.array(y)
    groups = np.array(groups)

    PA = np.transpose(X[np.where(y==1)])
    AP = np.transpose(X[np.where(y==2)])

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    axs.set_xlabel('Time (ms)')
    axs.set_ylabel('mV')
    if type(subject)==int:
        axs.set_title('MEP signals by Coil orientarion, subject ' + str(list_subjects[subject]) + ', groupnr: ' + str(subject) + ', trajs: ' + str(len(y)))
    # if no specific subject is specified
    else:
        axs.set_title('MEP signals by Coil orientarion' + ', total PAs: ' + str(len(np.transpose(PA))) + ', total APs: ' + str(len(np.transpose(AP))))

    if mean == True:
        axs.set_title('MEP signals by Coil orientarion, Mean')
        plt.plot(np.mean(PA, axis = 1), color = 'blue')
        plt.plot(np.mean(AP, axis = 1), color = 'red')
    else:
        plt.plot(PA, color = 'blue')
        plt.plot(AP, color = 'red')


    axs.legend(['PA','AP'], loc='best', title_fontsize='large', framealpha=0.5, facecolor='white', edgecolor='black', labelcolor=['blue','red'])

    plt.show()


def plot_subject_coil(X,y,list_subjects,groups,mean,subject):
    #plot a subject with PA and AP 
    #mean = True / False, set subject as int for wanted subject
    X = np.array(np.transpose(X))
    y = np.array(y)
    groups = np.array(groups)

    group_i_where=np.where(groups == subject)[0]
    yi = y[list(group_i_where)]
    Xi = X[list(group_i_where)]
    plot_coil(np.transpose(Xi),yi,list_subjects,groups,mean,subject)


def barplot(groups, mean_indi_scores, acc): #husk at tjek om onerow er slået til eller ej
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    axs.set_xlabel('Group number')
    axs.set_ylabel('Mean accuracy')
    acc = acc * 100
    axs.set_title(f'Mean accuracy pr subject, trained on X_latency, with overall mean accuracy: {acc:.2f}%')
    plt.bar(np.sort(list(set(groups))),mean_indi_scores)
    plt.show()


def PCA(X, explained = False, n=2, PCAs = True):# skal ind i plots
    from sklearn.decomposition import PCA
    if PCAs == True:
        # Create a PCA object with the desired number of components
        pca = PCA(n_components=n)

        # Fit the PCA model to the data and transform the data to the new space
        X_pca = pca.fit_transform(X)

        # The transformed data will now have two columns, which are the principal components
        # Create a scatter plot of the transformed data
        #plt.scatter(X_pca[:, 0], X_pca[:, 1])

        ''' # Add axis labels and a title
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA Scatter Plot')

        # Show the plot
        plt.show()'''
        
        plt.plot(X_pca[:, 0], label = "PC1")
        plt.plot(X_pca[:, 1], label = "PC2")
        plt.legend(loc="upper left")
        plt.title('Principal components')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.show()

        

    if explained == True:
        #Fit the PCA model to the data
        pca.fit(X)

        # Get the explained variance ratios
        variance_ratios = pca.explained_variance_ratio_

        # Create a bar plot of the explained variance ratios
        plt.bar(range(len(variance_ratios)), variance_ratios)

        # Add axis labels and a title
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio by Principal Component')

        # Show the plot
        plt.show()


if __name__ == "__main__":

    main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
    filelist = data.get_all_paths(main_path)
    X, y, groups, list_subjects = data.get_all_data(filelist)

    #what do you want to plot?
    #plot_groups(X, groups, list_subjects=list_subjects, specifics = [5,18,19,31,32])
    #plot_coil(X,y, list_subjects, groups, mean = False, subject = None)

    subject = 1 # set specific subject
    plot_subject_coil(X,y,list_subjects,groups,mean=False,subject=subject)


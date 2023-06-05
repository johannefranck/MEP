import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import data 
import random
import seaborn as sns
from sklearn import metrics
colorpalette = ['#2D3748','#738CB8', '#FCA311','#BFEDC1', '#F8F8F8']

def plot_groups(X, groups, list_subjects, specifics):
    # Plotting the different subject MEP signals in the same plot, with time on the x axis
    # set the specifics to ['all'] for all subjects otherwise specify: i.e. [3,13,23]
    sns.set_style("white")
    # Color pre-definitions
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # get a list of colors from the default color cycle in matplotlib
    # Create a dictionary with 50 unique color keys
    color_dict = {0: '#D59722', 1: '#ACE7FD', 2: '#D67220', 3: '#194D33', 4: '#48B233', 5: '#9900EF', 6: '#2F9100', 7: '#69D379', 8: '#B6F42F', 9: '#B0659C', 10: '#73830A', 11: '#0D48C2', 12: '#1B5AC3', 13: '#770159', 14: '#68C631', 15: '#877B85', 16: '#4D22B2', 17: '#EC36A1', 18: '#3D7957', 19: '#191D4D', 20: '#A696A3', 21: '#07A66F', 22: '#F4C01E', 23: '#556742', 24: '#2E3A01', 25: '#09AC36', 26: '#4638A2', 27: '#CF1B3B', 28: '#1BCF73', 29: '#3E9AB9', 30: '#EBAF15', 31: '#AB4147', 32: '#4CAF50', 33: '#EDBF24', 34: '#11BBAA', 35: '#F5C1A0', 36: '#42E978', 37: '#2D45B6', 38: '#AF7B0D', 39: '#F7FBD1', 40: '#A8E69B', 41: '#0F174F', 42: '#A62A92', 43: '#523368', 44: '#6E6708', 45: '#0A90CB', 46: '#EAB71B', 47: '#C5C765', 48: '#21329F', 49: '#142BB6'}

    # Get the time array with a sampling rate of 2000 Hz
    STAA = 12.5 # sliced_time_after_artifact, here it is 25 timepoints, which is the same a 12.5 ms
    time = np.arange(X.shape[0]) / 2000 * 1000 + STAA

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
    axs.legend(lines, legend_labels, loc='best', prop={'size': 'xx-small'}, title='Subject, nr', title_fontsize='small', framealpha=0.5, facecolor='white', edgecolor='black', labelcolor=colors)

    plt.show()



def plot_coil(X,y,list_subjects,groups, mean, subject):
    sns.set_style("white")
    #plotting coil orientations as PA (blue) and AP (red)
    #specify mean = TRUE if the mean of AP and PA is wanted, otherwise FALSE, subject is set to None
    X = np.array(np.transpose(X))
    y = np.array(y)
    groups = np.array(groups)

    PA = np.transpose(X[np.where(y==1)])
    AP = np.transpose(X[np.where(y==2)])

    STAA = 12.5  # sliced_time_after_artifact
    n_data_points = X.shape[1]
    time = np.arange(n_data_points) / 2000 * 1000 + STAA


    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    axs.set_xlabel('Time (ms)')
    axs.set_ylabel('mV')
    if type(subject)==int:
        axs.set_title('MEP by coil orientation, subject ' + str(list_subjects[subject])) #+ ', subject nr: ' + str(subject)) #+ ', trajs: ' + str(len(y)))
    # if no specific subject is specified
    else:
        axs.set_title('MEP signals by Coil orientation' + ', total PAs: ' + str(len(np.transpose(PA))) + ', total APs: ' + str(len(np.transpose(AP))))
    complementary_colors = ["#2D3748", "#FCA311"]

    if mean == True: #outcomment from here: see comment under. 
        axs.set_title('MEP signals by Coil orientation, Mean')
        plt.plot(time, np.mean(PA, axis = 1), color = complementary_colors[0],alpha=1)
        plt.plot(time, np.mean(AP, axis = 1), color = complementary_colors[1],alpha=1)
    else:
        plt.plot(time, PA, complementary_colors[0],alpha=1)
        plt.plot(time, AP, complementary_colors[1],alpha=1)

    fig.patch.set_facecolor('#F8F8F8')
    axs.set_facecolor('#F8F8F8')##EBEBEB

    axs.legend(['PA : '+str(len(np.transpose(PA))),'AP : '+str(len(np.transpose(AP)))], loc='best', title_fontsize='large', framealpha=0.5, facecolor='white', edgecolor='black', labelcolor=[complementary_colors[0],complementary_colors[1]])

    #plt.show()
    
    """#Outcomment this and comment above out, to plot where the maximum point is of the plot
    if mean == True:
        axs.set_title('MEP signals by Coil orientation, Mean')
        plt.plot(time, np.mean(PA, axis=1), color=complementary_colors[0], alpha=1)
        plt.plot(time, np.mean(AP, axis=1), color=complementary_colors[1], alpha=1)
    else:
        plt.plot(time, PA, complementary_colors[0], alpha=1)
        plt.plot(time, AP, complementary_colors[1], alpha=1)

    fig.patch.set_facecolor('#F8F8F8')
    axs.set_facecolor('#F8F8F8')

    axs.legend(['PA: ' + str(len(np.transpose(PA))), 'AP: ' + str(len(np.transpose(AP)))], loc='best',
               title_fontsize='large', framealpha=0.5, facecolor='white', edgecolor='black',
               labelcolor=[complementary_colors[0], complementary_colors[1]])

    # Find the positions of the maximum points along the feature length
    max_positions_PA = np.argmax(PA, axis=0)
    max_positions_AP = np.argmax(AP, axis=0)

    # Find the corresponding x-axis values
    x_axis_values = time

    # Plot a vertical line at the x-axis value of the maximum point for PA
    for pos in max_positions_PA:
        x_value = x_axis_values[pos]
        print(x_value)
        plt.axvline(x_value, color='blue', linestyle='--', alpha=0.5)

    # Plot a vertical line at the x-axis value of the maximum point for AP
    for pos in max_positions_AP:
        x_value = x_axis_values[pos]
        print(x_value)
        plt.axvline(x_value, color='red', linestyle='--', alpha=0.5)

    plt.show()"""



def plot_subject_coil(X,y,list_subjects,groups,mean,subject):
    sns.set_style("white")
    #plot a subject with PA and AP 
    #mean = True / False, set subject as int for wanted subject
    X = np.array(np.transpose(X))
    y = np.array(y)
    groups = np.array(groups)

    group_i_where=np.where(groups == subject)[0]
    yi = y[list(group_i_where)]
    Xi = X[list(group_i_where)]
    plot_coil(np.transpose(Xi),yi,list_subjects,groups,mean,subject)


def barplot(groups, mean_indi_scores, acc, xtype_title):
    sns.set_style("white")
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    
    fig.patch.set_facecolor('#F8F8F8')   # Set the background color for the figure
    axs.set_facecolor('#F8F8F8')          # Set the background color for the axes
    
    axs.set_xlabel('Group number')
    axs.set_ylabel('Mean accuracy')
    acc = acc * 100

    # Use the colorpalette for bar color
    title_switch = {'X': 'X', 'X_norm': 'X normalized', 'X_amplitude': 'Amplitude normalized', 'X_latency': 'Latency normalized', 'X_ampl_late': 'Only Amplitude and Latency'}

    axs.set_title(f'Mean accuracy per subject, with overall mean accuracy: {acc:.2f}% \n {title_switch[xtype_title]}')

    plt.bar(np.sort(list(set(groups))), mean_indi_scores, color='#738CB8')
    

def plot_accuracies(all_train_accuracies, all_val_accuracies, num_epochs):
    colorpalette = ['#2D3748','#738CB8', '#FCA311','#BFEDC1', '#F8F8F8']
    sns.set_style("white")
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    fig.patch.set_facecolor('#F8F8F8')  # Set the background color for the figure
    axs.set_facecolor('#F8F8F8')  # Set the background color for the axes

    axs.set_title(f"Training and validation accuracies mean over all subjects")
    axs.plot(range(1, num_epochs + 1), np.mean(all_train_accuracies, axis=0), label='Mean Training Accuracy', color=colorpalette[0])  # Using first color in palette
    axs.plot(range(1, num_epochs + 1), np.mean(all_val_accuracies, axis=0), label='Mean Validation Accuracy', color=colorpalette[1])  # Using second color in palette
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Accuracy')
    axs.legend()

    

def barplotmix(groups, mean_indi_scores_X, mean_indi_scores_X_norm, acc_x, acc_xnorm, xtype_title):
    colorpalette = ['#2D3748','#738CB8', '#FCA311','#BFEDC1', '#F8F8F8']
    sns.set_style("white")
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 5)) # increase plot size
    
    sorted_groups = np.sort(list(set(groups)))
    bar_width = 0.35 # specify the width of the bars
    
    # Subtract and add half the bar width to the x-values
    rects1 = axs.bar(sorted_groups - bar_width/2, mean_indi_scores_X, bar_width, alpha=0.5, color=colorpalette[0], label=f'X amplitude: {acc_x:.2f}%')
    rects2 = axs.bar(sorted_groups + bar_width/2, mean_indi_scores_X_norm, bar_width, alpha=0.5, color=colorpalette[1], label=f'X latency: {acc_xnorm:.2f}%')

    axs.set_xlabel('Subject number')
    axs.set_ylabel('Mean accuracy')
    
    if xtype_title in ['X and X normalized', 'X_norm', 'X_amplitude', 'X_latency', 'X amplitude and X latency']:
        axs.set_title(f'Mean accuracy pr subject\n {xtype_title}')

    # Move the legend to an empty space
    plt.legend(loc='lower left')
    # Set the background color to grey (#F8F8F8)
    axs.set_facecolor(colorpalette[-1])
    fig.set_facecolor(colorpalette[-1])

    #plt.savefig('barplot_mean_accuracy_pr_subject_Xampl_Xlat.png', dpi=300)

    #plt.show()#barplot_mean_accuracy_pr_subject_Xampl_Xlat


def PCA(X, explained = False, n=2, PCAs = True):# skal ind i plots
    sns.set_style("white")
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
        plt.xlabel('Sample points')
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

"""def confmat(y_test, predictions, title):
    cm = metrics.confusion_matrix(y_test, predictions)
    print(cm)  

    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap='RdBu')
    plt.title(title, size = 15) 
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["PA: 1", "AP: 2"], rotation=45, size = 10)
    plt.yticks(tick_marks, ["PA: 1", "AP: 2"], size = 10)
    plt.tight_layout()
    plt.ylabel('Actual label', size = 10)
    plt.xlabel('Predicted label', size = 10)
    width, height = cm.shape
    for x in range(width):
      for y in range(height):
        plt.annotate(str(cm[x][y]), xy=(y, x), 
        horizontalalignment='center',
        verticalalignment='center')
    plt.show()
"""
def confmat(y_test, predictions, title):
    cm = metrics.confusion_matrix(y_test, predictions)
    print(cm)

    color_palette = ['#425B87', '#4056A1', '#648EC0', '#8BB5E0', '#F1F1F1', '#FFFADE', '#FFEBA2', '#FFD775', '#FFC947', '#FCA311']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', color_palette)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix", size=15)
    plt.colorbar()
    plt.gca().set_facecolor('#F8F8F8')  # Set the background color
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["PA: 1", "AP: 2"], rotation=45, size=10)
    plt.yticks(tick_marks, ["PA: 1", "AP: 2"], size=10)
    plt.tight_layout()
    plt.ylabel('Actual label', size=15)
    plt.xlabel('Predicted label', size=15)
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center')
    


if __name__ == "__main__":
    import models
    main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
    filelist = data.get_all_paths(main_path)
    X, y, groups, list_subjects = data.get_all_data(filelist)
    X_norm = data.normalize_X(X, groups)
    X_amplitude, X_latency,X_ampl_late, X_diff,X_fft = data.other_X(X)
    """kfold cv
    # 10-fold stratified cross validation PR SUBJECT
    tot_scores_x, tot_indi_scores, mean_indi_scores_X, all_subject_coefficients = models.kfold_logisticregression_prsubject_stratified(X_amplitude, y, groups, onerow = True)
    # 10-fold stratified cross validation PR SUBJECT on X normalized
    tot_scores_xnorm, tot_indi_scores, mean_indi_scores_X_norm, all_subject_coefficients = models.kfold_logisticregression_prsubject_stratified(X_latency, y, groups, onerow = True)
    barplotmix(groups, mean_indi_scores_X, mean_indi_scores_X_norm, acc_x = np.mean(tot_scores_x), acc_xnorm = np.mean(tot_scores_xnorm), xtype_title = 'X amplitude and X latency')"""

    scores, mean_score_X, coefficients = models.logo_logreg_model(X_amplitude, y, groups, onerow =True)
    scores_Xnorm, mean_score_Xnorm, coefficients = models.logo_logreg_model(X_latency, y, groups, onerow =True)
    barplotmix(groups, scores, scores_Xnorm, acc_x = np.mean(mean_score_X), acc_xnorm = np.mean(mean_score_Xnorm), xtype_title = 'X amplitude and X latency')

    plt.savefig('barplot_mean_accuracy_LOGOCV_Xamplitude_Xlatency.png', dpi=300)
    
    #what do you want to plot?
    #plot_groups(X, groups, list_subjects=list_subjects, specifics = [5,18,19,31,32])
    #plot_coil(X,y, list_subjects, groups, mean = True, subject = None)
    #plot_subject_coil(X,y,list_subjects,groups,mean=False,subject=7)
    #plot_coil(X,y,list_subjects,groups, mean, subject)
    #for i in range(0,44):
    #    subject = i # set specific subject
    #    plot_subject_coil(X,y,list_subjects,groups,mean=False,subject=subject)
    #    plt.savefig(f'BaselineModel_Plots/MEPSubject{i}.png', dpi=100, bbox_inches='tight')
        #plt.show()
        #plt.close(fig)
        


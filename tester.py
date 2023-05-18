import matplotlib.pyplot as plt
import numpy as np
import data
from pymatreader import read_mat
import models



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for your plot using seaborn
sns.set_style("darkgrid")

# Generate some sample data for plotting (replace this with your own data)
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data with a line plot
ax.plot(x, y, label='Sine curve', linewidth=2, color='blue', linestyle='-')

# Set the plot title and axis labels
ax.set_title('Professional Plot Example', fontsize=18, fontweight='bold', pad=15)
ax.set_xlabel('X-axis Label', fontsize=14, fontweight='bold')
ax.set_ylabel('Y-axis Label', fontsize=14, fontweight='bold')

# Customize tick labels
ax.tick_params(axis='both', labelsize=12)

# Add grid lines
ax.grid(True)

# Add a legend
ax.legend(loc='best', fontsize=12)

# Remove the top and right spines
sns.despine()


# Show the plot
plt.show()




if __name__ == '__main__':
    main_path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"
    filelist = data.get_all_paths(main_path)
    X, y, groups, list_subjects = data.get_all_data(filelist)

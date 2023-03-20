'''
import os

# Define the path
path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"

# Initialize a list to store the unique subjects (to use as groups)
groups = []

i = 0
# Loop over all files in the directory
for root, dirs, files in os.walk(path):
    for file in files:
        # Check if the filename contains sub
        if "sub" in file:
            
            filepath = path + str('/') + file
            # Extract the tag and add it to the list of unique tags
            subject = filepath[39:45]
            
            groups.append(i)
    i = i + 1

# Print the list of unique tags
print(groups)



import os

# Define the path
path = "/mnt/projects/USS_MEP/COIL_ORIENTATION"

# Initialize a list to store the unique tags
unique_tags = []

# Loop over all files in the directory
for root, dirs, files in os.walk(path):
    for file in files:
        # Check if the filename contains the tag "X99909"
        if "X99909" in file:
            # Extract the tag and add it to the list of unique tags
            tag = "X99909"
            if tag not in unique_tags:
                unique_tags.append(tag)

# Print the list of unique tags
print(unique_tags)

'''



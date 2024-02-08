# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set()

# requirements file to make: pip freeze > requirements.txt
# file names: BBox_List_2017, Data_Entry_2017

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

# print(os.listdir('./archive'))

filename = "Data_Entry_2017"

# Load the dataset into a pandas DataFrame
df = pd.read_csv('./archive/{}.csv'.format(filename))

# The 'Finding Labels' column contains multiple labels separated by '|', so we need to split these labels
# and then explode the DataFrame to have one label per row.
df_labels = df['Finding Labels'].str.split('|').explode()

# Now we count the occurrences of each label
label_counts = df_labels.value_counts()

# get finding labels and count separately from label_counts
finding_labels = label_counts.index
count = label_counts.values

print(finding_labels)

sns.barplot(label_counts, color='b')
plt.title('Distribution of Classes for Training Dataset', fontsize=15)
plt.xlabel('Number of Patients', fontsize=15)
plt.ylabel('Diseases', fontsize=15)
plt.show()

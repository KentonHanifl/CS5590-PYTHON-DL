from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")


headers = open("C:/Users/Kenton/Desktop/School/PythonDeepLearning/Python_Lesson5/data/College.csv").readline().split(',')

i = 0
while i < len(headers):
    headers[i] = headers[i].strip('"').strip('"\n')
    i += 1
headers = headers[2:]
header1 = 2
header2 = 7

dataset = pd.read_csv('C:/Users/Kenton/Desktop/School/PythonDeepLearning/Python_Lesson5/data/College.csv')


x = dataset.iloc[:,[header1,header2]]

##x = preprocessing.normalize(x)
##
##scaler = preprocessing.StandardScaler()
##
##scaler.fit(x)
##X_scaled_array = scaler.transform(x)
##X_scaled = pd.DataFrame(X_scaled_array, columns = ['Apps','F.Undergrad'])
##
##X_scaled = np.square(X_scaled)
##X_scaled = 10*X_scaled
nclusters = 2 # this is the k in kmeans
seed = 0

mdl = KMeans(n_clusters=nclusters, random_state=seed).fit(x)

# predict the cluster for each data point
y_cluster_kmeans = mdl.predict(x)

d = pd.DataFrame(y_cluster_kmeans)
##print(X_scaled)

import numpy as np
import matplotlib.pyplot as plt

# Create plot
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)#, axisbg="1.0")
LABEL_COLOR_MAP = {0 : 'red',
                   1 : 'green',
                   2 : 'black',
                   3 : 'blue',
                   4 : 'yellow'
                   }

color = [LABEL_COLOR_MAP[i] for i in d[0]]
plt.scatter(x["Apps"], x["F.Undergrad"], alpha=0.8, c=color)
 
plt.title('Matplot scatter plot')
plt.show()

wcss = []
##elbow method to know the number of clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

---
title: "PCA analysis and clustering images"

image: /images/cover.jpg
categories:
  - Classification
tags:
  - content
  - image Classification
  - convolutional neural Network
  - CNN
last_modified_at: 2021-03-17T10:46:49-04:00

---
This project clusters images using K-means clustering algorithm. The dimensionality of the images are reduced by Principal component analysis (PCA).

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.cluster import KMeans
from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.preprocessing import StandardScaler
from time import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```

# PCA of the images


```python
data_path = r'/jet/home/mhrahman/Projects/HW1/HW1/Classification'
folders = os.listdir(data_path)
arrays = []
label_true = []
for i,k in enumerate(folders):
    dir_name = os.path.join(data_path,k)
    os.chdir(dir_name)
    images = os.listdir(dir_name)
    for j in images[:50]:
        a = np.array(Image.open(j).resize((80,128)).convert('L')).flatten().reshape(1,-1)[0]
        arrays.append(a)
        label_true.append(i)
```


```python
#sc = StandardScaler()
#sc.fit(arrays)
#ar = sc.transform(arrays)
```


```python
mat = np.asmatrix(arrays)
mat.shape
```




    (100, 10240)




```python
pca = PCA(100).fit(mat)
exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
plt.figure(figsize=(10,6))
plt.bar(range(1,len(exp_var_pca)+1), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1,len(cum_sum_eigenvalues)+1), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio',fontsize = 16)
plt.xlabel('Principal component index',fontsize = 16)
plt.xticks(np.arange(1, 101, step=3))
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('/jet/home/mhrahman/Projects/HW3/Figure/PCA_variance_main.jpg',dpi = 300)
plt.show()
```



![image-center]({{ site.url }}{{ site.baseurl }}/images\PCA analysis and clustering images\output_5_0.png){: .align-center}




```python
plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(0, 100, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 100, step=3)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.savefig('/jet/home/mhrahman/Projects/HW3/Figure/PCA_variance.jpg',dpi = 300)
plt.show()
```



![image-center]({{ site.url }}{{ site.baseurl }}/images\PCA analysis and clustering images\output_6_0.png){: .align-center}



# Reconstruction of image


```python
image_path = r'/jet/home/mhrahman/Projects/HW1/HW1/Classification/pre-CHF'
os.chdir(image_path)
image = os.listdir(image_path)[0]
#Image.open(image,'r').convert('L')
```


```python
my_image = imread(image)
print(my_image.shape)
plt.figure(figsize=[12,8])
plt.imshow(my_image, cmap='Greys_r')
plt.savefig('/jet/home/mhrahman/Projects/HW3/Figure/original.jpg',dpi = 300)
```

    (800, 1280)





    <matplotlib.image.AxesImage at 0x7efcfed6e090>





![image-center]({{ site.url }}{{ site.baseurl }}/images\PCA analysis and clustering images\output_9_2.png){: .align-center}




```python
def reconstruct(image,n_components):
    img_array = np.array(np.array(Image.open(image,'r').convert('L'))/255)
    pca_img = PCA(n_components).fit(img_array)
    trans_pca = pca_img.transform(img_array)
    img_arr = pca_img.inverse_transform(trans_pca)
    return plt.imshow(img_arr, cmap='Greys_r')
```


```python
ks = [2, 10, 25, 50, 100, 150]
plt.figure(figsize=[15,9])
for i in range(len(ks)):
    plt.subplot(2,3,i+1)
    reconstruct(image,ks[i])
    plt.title("Components: "+ str(ks[i]))

plt.subplots_adjust(wspace=0.2, hspace=0.0)
plt.savefig('/jet/home/mhrahman/Projects/HW3/Figure/reconstructed.jpg',dpi = 300)
plt.show()
```



![image-center]({{ site.url }}{{ site.baseurl }}/images\PCA analysis and clustering images\output_11_0.png){: .align-center}



#  Error of the reconstructed images


```python
img_array = np.array(np.array(Image.open(image,'r').convert('L'))/255)
start = 0
max_components = 50
error = []
for i in range(start,max_components):
    pca = PCA(n_components=i)
    pca2 = pca.fit_transform(img_array)
    pca2_project = pca.inverse_transform(pca2)
    total_loss = np.linalg.norm(img_array - pca2_project)
    error.append(total_loss)

plt.clf()
plt.figure(figsize=(10,6))
plt.title("Reconstruct error of pca",fontsize = 18)
plt.plot(error,'r')
plt.xticks(range(len(error)), range(start,max_components), rotation='vertical')
plt.xlim([-1, len(error)])
plt.xlabel("Number of PCs",fontsize = 16)
plt.ylabel("Square error",fontsize = 16)
plt.savefig('/jet/home/mhrahman/Projects/HW3/Figure/error.jpg',dpi = 300)
plt.show()
```


    <Figure size 432x288 with 0 Axes>




![image-center]({{ site.url }}{{ site.baseurl }}/images\PCA analysis and clustering images\output_13_1.png){: .align-center}



# Cluster analysis


```python
def cluster(n_component,mat):
    t1 = time()
    reduced_data = PCA(n_components=n_component).fit_transform(mat)
    kmeans = KMeans(n_clusters=2,init='k-means++',n_init=10)
    kmeans.fit(reduced_data)
    t2 = time()
    elapsed_time = t2- t1
    centers = kmeans.cluster_centers_
    label_predicted = kmeans.fit_predict(reduced_data)
    return reduced_data,label_predicted,elapsed_time
```


```python
PCs = [2,10,20,40,60,80]
adjust_score = []
silh_score = []
times = []
for i in PCs:
    PCA_conv,label_predicted,elaspsed_time = cluster(i,mat)
    ad_score = adjusted_rand_score(label_predicted,label_true)
    sil_score = silhouette_score(PCA_conv,label_predicted)
    adjust_score.append(ad_score)
    silh_score.append(sil_score)
    times.append(elaspsed_time)
```


```python
df = pd.DataFrame({'PCs':PCs,'Adjuster_rand_score':adjust_score,'Silhouette Score':silh_score,'Time(sec)': times})
df.to_csv('/jet/home/mhrahman/Projects/HW3/Figure/Score.csv',index = False)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PCs</th>
      <th>Adjuster_rand_score</th>
      <th>Silhouette Score</th>
      <th>Time(sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.736995</td>
      <td>0.363978</td>
      <td>17.422835</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>0.457481</td>
      <td>0.200441</td>
      <td>21.196272</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>0.882417</td>
      <td>0.143811</td>
      <td>15.999109</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40</td>
      <td>0.920801</td>
      <td>0.117854</td>
      <td>21.099517</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>0.669167</td>
      <td>0.119366</td>
      <td>15.299866</td>
    </tr>
    <tr>
      <th>5</th>
      <td>80</td>
      <td>0.920801</td>
      <td>0.100475</td>
      <td>25.699217</td>
    </tr>
  </tbody>
</table>
</div>




```python
reduced_data = PCA(n_components=80).fit_transform(mat)
kmeans = KMeans(n_clusters=2,init='k-means++',n_init=10)
kmeans.fit(reduced_data)
centers = kmeans.cluster_centers_
label_predicted = kmeans.fit_predict(reduced_data)
```


```python
plt.figure(figsize=(12,12))
fig = plt.figure()
ax = fig.add_subplot(111,projection ='3d')
scatter = ax.scatter(centers[:,1],centers[:,0],centers[:,2], s = 250, marker = 'o',c = 'red', label = 'centroid')
scatter = ax.scatter(reduced_data[:,1],reduced_data[:,0],reduced_data[:,2],c = label_predicted.astype(np.float64),s =20,cmap ='winter')
ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')
ax.set_zlabel('Principal component 3')
ax.legend()
plt.savefig('/jet/home/mhrahman/Projects/HW3/Figure/Cluster.jpg',dpi = 300)
plt.show()
```


    <Figure size 864x864 with 0 Axes>




![image-center]({{ site.url }}{{ site.baseurl }}/images\PCA analysis and clustering images\output_19_1.png){: .align-center}




```python

```

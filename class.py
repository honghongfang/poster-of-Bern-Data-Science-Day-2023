from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os.path
import seaborn as sns 
sns.set()
import contextlib
from seaborn import _statistics 
from matplotlib.colors import LogNorm, Normalize
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from numpy.linalg import norm
import pickle
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from scipy.linalg import inv
import glob

#class InputImage:
    
class PreClassification:
    def __init__(self, bin_count=512, min_count_threshold=10, 
                       threshold_rise_frac=0, sigma=1,row=2,cloumn=3,size=12, psize=5,
                       scale=6, take_n=50000):
        self.bin_count = bin_count
        self.min_count_threshold = min_count_threshold
        self.threshold_rise_frac = threshold_rise_frac
        self.sigma = sigma
        self.row = row
        self.cloumn = cloumn
        self.size = size
        self.psize = psize
        self.scale = scale
        self.take_n = take_n

    def tiffread(self, path):
        """
        The number of frames of the multidimensional image
        Args:
            path : image path
            
        Returns:
            Array form of all frames of this image
        """
        img = Image.open(path)
        images = []
        for i in range(self.img.n_frames):
            self.img.seek(i)
            self.images.append(np.array(self.img))
        return np.array(self.images)
    
    def get_signal_threshold(self, data):
        """
        Assuming gaussian distribited background, obtain signal threshold
        Args:
            data (iterable): data array
            bin_count (int): number of bins in histogram
            min_count_threshold (int): min number of values in a bin for 
                                       estimation of the minimum background value
            threshold_rise_frac (float): Fraction of the half width of bg distribution by which hreshold will be inreased.
            
        Returns:
            signal threshold value
        """
        counts, bins = np.histogram(data, bins=self.bin_count)
        bin_mid = (self.bins[:-1]+self.bins[1:]) / 2

        highest_bin_idx = self.counts.argmax()
        val_highest_bin = self.bin_mid[self.highest_bin_idx]
        val_smallest_nonzero = self.bin_mid[0]

        first_bin_above_thresholt_idx = np.argmax(self.counts>self.min_count_threshold)
        val_first_bin_above_thresholt = self.bin_mid[self.first_bin_above_thresholt_idx]

        sinal_threshold = self.val_highest_bin + (self.val_highest_bin - self.val_first_bin_above_thresholt) * (1+self.threshold_rise_frac)
        
        return self.sinal_threshold
        
    def get_smooth_img(self, image):
        """
        Smooth multi-frame image using Gaussian function
        Args:
            image : image array after stack
            
        Returns:
            smoothed image
        """
        smooth_img = np.empty(image.shape)
        for i in range(len(image)):
            smooth_img[i] = gaussian(image[i], sigma=self.sigma, preserve_range=True)
        return smooth_img 
    
    def subplots(self, image,image_name):
        """
        Image comparison of four channels
        Args:
            image : image array after stack
            
        Returns:
            Histogram comparison of six images
        """
        fig, ax = plt.subplots(nrows=self.row, ncols=self.cloumn, figsize=(self.size * 3, self.size * 2))
        for r in range(self.row):
            for c in range(self.cloumn):
                if r == 0:
                    ax[r,c].hist2d(x=image[:,r],y=image[:,c+1], bins=1000, norm=LogNorm(), cmap='jet');
                    ax[r,c].set_xlabel('g', fontsize=40)
                    ax[r,0].set_ylabel('r', fontsize=40)
                    ax[r,1].set_ylabel('f', fontsize=40)
                    ax[r,2].set_ylabel('b', fontsize=40)
                elif r == 1 and c != 2:
                    ax[r,c].hist2d(x=image[:,r],y=image[:,c+2], bins=1000, norm=LogNorm(), cmap='jet');
                    ax[r,c].set_xlabel('r', fontsize=40)
                    ax[r,0].set_ylabel('f', fontsize=40)
                    ax[r,1].set_ylabel('b', fontsize=40)
                else:
                    ax[r,c].hist2d(x=image[:,r+1],y=image[:,c+1], bins=1000, norm=LogNorm(), cmap='jet');
                    ax[r,c].set_ylabel('b', fontsize=40)
                    ax[r,c].set_xlabel('f', fontsize=40)
         plt.savefig('{}'.format(image_name))
    
    def subplots_scatter(self, image,image_name):
        """
        Image comparison of four channels
        Args:
            image : image array after stack 
            color : label
        Returns:
            Scatter plot comparison of six images
        """
        cmap = 'jet'
        fig, ax = plt.subplots(nrows=self.row, ncols=self.cloumn, figsize=(self.size * 3, self.size * 2))
        for r in range(self.row):
            for c in range(self.cloumn):
                if r == 0:
                    ax[r,c].scatter(x=image[:,r],y=image[:,c+1], c=color, s=self.psize, cmap=cmap);
                    ax[r,c].set_xlabel('g', fontsize=40)
                    ax[r,0].set_ylabel('r', fontsize=40)
                    ax[r,1].set_ylabel('f', fontsize=40)
                    ax[r,2].set_ylabel('b', fontsize=40)
                elif r == 1 and c != 2:
                    ax[r,c].scatter(x=image[:,r],y=image[:,c+2], c=color, s=elf.psize, cmap=cmap);
                    ax[r,c].set_xlabel('r', fontsize=40)
                    ax[r,0].set_ylabel('f', fontsize=40)
                    ax[r,1].set_ylabel('b', fontsize=40)
                else:
                    ax[r,c].scatter(x=image[:,r+1],y=image[:,c+1], c=color, s=self.psize, cmap=cmap);
                    ax[r,c].set_ylabel('b', fontsize=40)
                    ax[r,c].set_xlabel('f', fontsize=40)
         plt.savefig('{}'.format(image_name))
         
      def show_images_mip(self, ims,ims_name):
        """
        Multi-channel multi-frame image visualization
        Args:
            image : list of tiff image 
        Returns:
            image
        """
        n = len(ims)
        fig, ax = plt.subplots(1, n, figsize=(self.scale*n, self.scale*1))
        for ax_i, im_i in zip(self.ax, ims):
            self.ax_i.grid(False)
            self.ax_i.imshow(self.im_i.max(axis=0), cmap='gray')
         plt.show()
         plt.savefig('{}'.format(ims_name))
         plt.close()
         
    def subsample_rad(self, ds):
        """
        This is sampled with p ~ exp(r^4)
        Args:
            ds : Original image data array format 
        Returns:
            Image data after sampling
        """
      n = len(ds)
      if type(self.take_n)!= int:
        self.take_n = int(self.take_n*n)
      max = ds.max(axis=0, keepdims=True)
      mask = (ds != max).all(axis=1)
      ds_nonmax = ds[self.mask]
      n_nonmax = len(self.ds_nonmax)
      mean = self.ds_nonmax.mean(axis=0, keepdims=True)
      ds_nonmax -= self.mean
      #ds_nonmax -= 50
      #print(ds_nonmax.shape)
      ds_nonmax = self.ds_nonmax.clip(0)
      mask = (self.ds_nonmax != 0).all(axis=1)
      ds_nonmax = self.ds_nonmax[mask]
      n_nonmax = len(self.ds_nonmax)
      #print(ds_nonmax.shape)
      range = np.percentile(self.ds_nonmax, 99.999, axis=0, keepdims=True)
      #print(n, n_nonmax, range, max, mean)
      range /=2
      ds_scaled = (self.ds_nonmax/range).clip(0, 1)
      ds_r  = np.sqrt((self.ds_scaled**2).sum(axis=1))
      #plt.hist(ds_r, log=True)
      #plt.show()
      p = np.exp(self.ds_r**4)
      p /= self.p.sum()
      #plt.semilogy(ds_r, p, '.')
      #plt.show()
      ds_idx = np.random.choice(self.n_nonmax, self.take_n, replace=False, p=self.p)
      ds_sub = self.ds_nonmax[self.ds_idx]
      ds_sub_r  = np.sqrt((self.ds_sub**2).sum(axis=1))
      #plt.hist(ds_sub_r, log=True)
      #plt.show()
      return self.ds_sub

class K_Means:
    def __init__(self, n_cluster, epochs=10):
        self.n_cluster = n_cluster
        self.epochs = epochs
        pass
    def init_centers(self, X):
    # idx = np.random.randint(len(X), size=(self.n_cluster,))
    # centers = X[idx,:]
        centers = np.eye(self.n_cluster, self.n_cluster)
        return centers

    @staticmethod
    def norm(arr):
        return arr / np.sqrt((arr**2).sum(axis=1, keepdims=True))

    @staticmethod
    def calculate_distance(arr1,arr2,norm=True):
        if len(arr1.shape)==1:
            arr1 = arr1[np.newaxis]
        if len(arr2.shape)==1:
            arr2 = arr2[np.newaxis]
        arr1_norm = K_Means.norm(arr1) if norm else arr1
        arr2_norm = K_Means.norm(arr2) if norm else arr2

        similarity = np.einsum('ij,kj->ik', arr1_norm,arr2_norm)
        distance = 1 - similarity

        return distance

    def update_centers(self, X):
        predict_class = self.predict(X, norm=False)

        centers = self.norm(self.centers)
        for ct in range(len(centers)):
            idx = predict_class == ct

            samples = X[idx, :]

            assert len(samples)>0
            centers[ct] = np.mean(samples,axis=0)
        centers = self.norm(centers)
        self.centers = centers
        return self.centers

    def fit(self, X, y=None):
        X_norm = K_Means.norm(X)
        self.center_history = []
        self.centers = self.init_centers(X_norm)
        for epoch in range(self.epochs):
            self.centers = self.update_centers(X_norm)
            self.center_history.append(self.centers.copy())
        return self.centers

    def predict(self, X, norm=True):
        centers = self.centers
        distance = K_Means.calculate_distance(X,centers, norm=norm)
        min_distance_idx = np.argmin(distance,axis = 1)
        predict_class = min_distance_idx
        return predict_class
    def score(self, X):
        pass
        
class F(tf.keras.layers.Layer):
  def __init__(self, n_dim, **kwargs):
    super().__init__(**kwargs)
    self.n_dim = n_dim

  def build(self, input_shape):
    self.p = self.add_weight(initializer=tf.keras.initializers.Constant(0.), shape=(self.n_dim,), name='p',dtype=tf.float32)
    super().build(input_shape)

  def call(self, x):
    dist_lines=[]
    for i in range(self.n_dim):
      c_i = x[..., i, 0, :]
      v_i = x[..., i, 1, :]

      d_i = point_distance_line(self.p,c_i,v_i)

      dist_lines.append(d_i)
    d = tf.reduce_sum(dist_lines)

    return d  
    
 class GradientDescent: 
    def __init__(self, ,n_dim=4,learning_rate=10,epochs=100,batch_size = 1):
        self.n_dim = n_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
       
    def point_distance_line(self, point,center,vector):
      """
      distance from point to line in space
      """
      CP = point - center
      CP_dot_V_over_len_V = tf.reduce_sum(self.CP*vector) / tf.reduce_sum(vector*vector)
      CN = self.CP_dot_V_over_len_V*vector
      NP = self.CP - self.CN
      distance = tf.sqrt(tf.reduce_sum(self.NP * self.NP))
      return self.distance
    @staticmethod
    def loss_f(true_y, y):
      """
      cost function
      """
      return y

    def get_trainable_variables(self, cluster_axes,cluster_means):
      """
      In the framework of TensorFlow, the gradient descent method is used to
      find the training variable, namely the origin of the new coordinate system
      """
      x = tf.keras.layers.Input(name='x', dtype=tf.float32, shape=(self.n_dim,2,self.n_dim))
      # shape=(4, 2, 4)  # 4 lines, point on the line & line vector, 4 components of the vectors
      f_layer = F(self.n_dim=self.n_dim)
      f = f_layer(self.x)
      model = tf.keras.Model(inputs=self.x, outputs=self.f)
      optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
      model.compile(optimizer=self.optimizer, loss=loss_f)
      #model.summary()
      centers_and_components = np.stack((cluster_means, cluster_axes), axis=1)[np.newaxis]
      hist = model.fit(x=self.centers_and_components, y=np.array([0]), batch_size=self.batch_size, epochs=epochs, verbose=None)
      t = model.trainable_variables[0].numpy()
      #plt.plot(hist.history['loss'])
      return t
  

    
class UnMix:
    def __init__(self, channels = 4,frames = 26,size_row = 574,size_column = 574):
        self.channels = channels
        self.frames = frames
        self.size_row = size_row
        self.size_column = size_column
    
    def unmix_img(self, new_centers,image_s_f):
    """
    The transformation matrix U is used to transform the image to the new coordinates
    """
      U = inv(new_centers.T)
      unmix = np.dot(self.U,np.array(image_s_f))
      unmix_positive = self.unmix.clip(min=0)
      array_unmix_positive = self.unmix_positive.reshape(self.channels,self.frames,self.size_row,self.size_row).astype(np.uint16)
      return array_unmix_positive
      
    def produce_img(self, array,time):
    """
    Generate a new multi-channel multi-frame image in a new coordinate system
    """
        for i in range(array.shape[0]):
            a = array[i]
            imlist = []
            for m in a:
                imlist.append(Image.fromarray(m))
            imlist[0].save("{}_{}.tif".format(time,i), compression="tiff_deflate", save_all=True,append_images=imlist[1:])
        
def eachFile(filepath):
  pathDir =  os.listdir(filepath)
  for allDir in pathDir:
    child = os.path.join(filepath, allDir)
    if os.path.isfile(child):
      readFile(child)
      continue
    eachFile(child)
# Read the files with the same suffix in the specified path 
def readFile(filenames):
  if '.tif' in filenames:
    PATH.append(filenames)
filenames = "data/2PM data/Josy_trafficking/20220715_EK_MV_SG_JM/220715_Doc1_15-00-08__pos_1"  # refer root dir
PATH =[]
eachFile(filenames)

def same_time_image(PATH,time = "0000"):
  path = []
  for i in range(len(PATH)):
    if PATH[i][-12:-8] == time:
        path.append(tiffread(PATH[i]))
  return path
time = "0000"
ims = same_time_image(PATH,time = time)
ims_smooth = [get_smooth_img(im_i) for im_i in ims]
ims_smooth_f = [im_i.flatten() for im_i in ims_smooth]
mask_s = [im_i>get_signal_threshold(im_i, threshold_rise_frac=5) for im_i in ims_smooth_f]
mask_s = np.any(mask_s, axis=0)
ds = np.stack([im_i[mask_s] for im_i in ims_smooth_f], axis=-1)
show_images_mip(ims)
show_images_mip(ims_smooth)
show_images_mip([im_i.clip(min=500) for im_i in ims_smooth]) # Pixels smaller than three hundred are displayed as three hundred
subplots(ds)
ds_sub_r = subsample_rad(ds)
subplots(ds_sub_r)
kmeans = K_Means(4, epochs=30)
centers = kmeans.fit(ds_sub_r)  
cluster_labels_sub=kmeans.predict(ds_sub_r) # sub data
cluster_labels=kmeans.predict(ds) # all data
subplots_scatter(ds_sub_r,cluster_labels_sub)
subplots_scatter(K_Means.norm(ds_sub_r), cluster_labels_sub)
#PCA
labels = [cluster_labels_sub == 0,cluster_labels_sub == 1,cluster_labels_sub == 2,cluster_labels_sub == 3]
ds_num = [ds_sub_r[label] for label in labels]
pca_num = [PCA(n_components=1) for i in range(4)]
cluster_axes = [p.fit(d).components_[0] for p in pca_num for d in ds_num][:4]
cluster_means = [d.mean(axis=0)for d in ds_num]
t = get_trainable_variables(cluster_axes,cluster_means)
new_ds_sub_r = ds_sub_r - t
new_ds = ds - t
new_kmeans = K_Means(4, epochs=30)
new_centers = new_kmeans.fit(new_ds_sub_r)
new_cluster_labels_sub = new_kmeans.predict(new_ds_sub_r)
new_cluster_labels=kmeans.predict(new_ds)
subplots_scatter(new_ds_sub_r,new_cluster_labels_sub)
array_unmix_positive = unmix_img(new_centers,ims_smooth_f)
produce_img(array_unmix_positive,time = time)
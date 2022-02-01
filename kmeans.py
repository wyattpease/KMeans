import numpy as np

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):

    p = generator.randint(0, n) #this is the index of the first center
    
    r = generator.rand()
    centers = [None] * n_cluster
    centers[0] = p
    temp_list = np.copy(x)
    for i in range(1,n_cluster):
        temp = 0
        c = -1
        centroid_dist = []
        temp_list = x
        for j in range(i):
            last_centroid=x[centers[j]]
            sub = np.subtract(temp_list,last_centroid)
            sq = np.square(sub)
            sum = np.sum(sq, axis=1)
            distances = np.sqrt(sum)
            if len(centroid_dist)==0:
                centroid_dist=sum
            else:
                centroid_dist = np.column_stack((centroid_dist,sum))

        dist_min = []
        if centroid_dist.ndim == 1:
            dist_min = centroid_dist
        else:
            dist_min = np.min(centroid_dist,axis=1)
        dist_sum = np.sum(dist_min)
        dd = np.divide(dist_min,dist_sum)
        total = 0.0
        for j in range(len(dist_min)):
            total += dd[j]
            if total > r:
                centers[i] = j
                break

    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        
        centroids = np.take(x,self.centers,axis=0)
        self.centers = centroids
        for i in range(self.max_iter):
            
            
            centroid_distances = []

            for j in range(self.n_cluster):
                curr_centroid=self.centers[j]
                sub = np.subtract(x,curr_centroid)
                sq = np.power(sub,2)
                sum = np.sum(sq,axis=1)
                centroid_distances.append(sum)
            centroid_distances = np.array(centroid_distances)

            #get clusters with min dist frome each point
            min_dist = np.argmin(centroid_distances, axis=0)
            #update centroids
            old_centroids = self.centers
            new_centers = []
            for i in range(self.n_cluster):
                cluster_members = np.where(min_dist == i)
                points = np.take(x, cluster_members, axis=0)
                points.reshape((points.shape[1:]))
                pts_sum = np.sum(points, axis=1)
                new_centroid =np.divide(pts_sum,len(cluster_members[0]))
                new_centers.append(new_centroid[0])
            new_centers = np.array(new_centers)
            sub = np.subtract(new_centers,old_centroids)
            sq = np.square(sub)
            sum = np.sum(sq)
            total_e = np.divide(sum, self.n_cluster)
            if self.e > total_e:
                return (new_centers,min_dist,i)
            else:
                self.centers = new_centers

        return (self.centers,min_dist,self.max_iter)
                
        


class KMeansClassifier():

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape

        km = KMeans(self.n_cluster)
        self.centroids,membership,i = km.fit(x, centroid_func)
        centroid_labels = np.zeros(self.n_cluster)
        cluster_by_label = np.column_stack((membership, y))

        for i in range(self.n_cluster):
            cluster_members = cluster_by_label[np.where(cluster_by_label[:,0]==i)]
            (unique, counts) = np.unique(cluster_members[:,1], return_counts=True)
            centroid_labels[i] = unique[np.argmax(counts)]

        self.centroid_labels = centroid_labels

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape

        predicted_labels=[]
        for i in range(N):
            sub = np.subtract(self.centroids,x[i])
            sq = np.power(sub,2)
            sum = np.sum(sq,axis=1)
            c = np.argmin(sum)
            predicted_labels.append(self.centroid_labels[c])

        return np.array(predicted_labels)



def transform_image(image, code_vectors):

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    t_im = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sub = np.subtract(code_vectors,image[i][j])
            sq = np.power(sub,2)
            sum = np.sum(sq,axis=1)
            c = np.argmin(sum)
            t_im[i][j] = code_vectors[c]

    return t_im
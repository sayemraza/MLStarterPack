class k_means:
  def __init__(self,x,k):
    self.x = x
    self.k = k
  
  def get_cluster(self,centroids):
    centroid_assignment = {}
    for i in range(self.x.shape[0]):
        dist_from_centroid = []
        for j in range(self.k):
            dist = np.linalg.norm(self.x[i,:]-centroids[j,:])
            dist_from_centroid.append(dist)
        centroid_assignment[i] = np.argmin(dist_from_centroid)
    return centroid_assignment
    
  def create_cluster(self):
    centroids = np.mat(np.zeros((self.k,self.x.shape[1])))
    for j in range(self.x.shape[1]):
        min_j = min(self.x[:,j])
        range_j = float(max(self.x[:,j]) - min_j)
        centroids[:,j] = min_j + range_j * np.random.rand(self.k, 1)
    centroid_assignment = get_cluster(self.x,centroids)
    cluster_change = True
    while cluster_change:
        old_cent = centroid_assignment.copy()
        C=[]
        for i in range(self.k):
            points = [self.x[j] for j in range(self.x.shape[0]) if centroid_assignment[j] == i]
            C.append(list(np.mean(points, axis=0)))
        centroids = np.array(C)
        centroid_assignment = self.get_cluster(centroids)
        if centroid_assignment==old_cent:
            cluster_change = False
    return centroid_assignment,centroids

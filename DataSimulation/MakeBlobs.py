# Use sklearn to create simulated dataset for use with clustering

# Load library
from sklearn.datasets import make_blobs

# Generate feature matrix and target vector
features, target = make_blobs(n_samples = 100,
                                                 n_features = 2,
                                                 centers = 3,
                                                 cluster_std = 0.5,
                                                 shuffle = True,
                                                 random_state = 1)

# View feature matrix and target vector
print("Feature Matrix\n", features[:3])
print("Target vector\n", target[:3])
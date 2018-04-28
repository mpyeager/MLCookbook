# Template for importing toy datasets for algo exploration and testing

# Load sklearn's datasets
from sklearn import datasets

# To load [sample] dataset below, uncomment desired dataset
# Comment out datasets not being tested, used
digits = datasets.load_digits() #1,797 images of handwritten digits, good for teaching image classification 
boston = datasets.load_boston() #503 Boston housing prices, good for exploring regression algos
iris = datasets.load_iris() #150 measurements of Iris flowers, good for exploring classification algos

# Create features matrix by replacing [dataset] with desired dataset from load above
features = [dataset].data

# Create target vector by replacing [dataset] with desired dataset from load above
target = [dataset].target

# View first observation
features[0]
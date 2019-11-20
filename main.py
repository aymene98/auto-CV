from model import Model
from load_data import load_cifar, load_mnist, load_PA100K, load_PA100K_10
import time


print("Loading MNIST ...")
x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = load_mnist()
print("Done")

# Training for the mnist dataset.
print("**************** Training on MNIST *****************")
model = Model()
model.train(x_train_mnist,y_train_mnist)
_ = model.test(x_test_mnist,y_test_mnist)

# To empty the RAM
del x_train_mnist
del y_train_mnist
del x_test_mnist
del y_test_mnist
del model

time.sleep(5)

print("Loading CIFAR-100 ...")
x_train_cifar, y_train_cifar, x_test_cifar, y_test_cifar = load_cifar()
print("Done")

# Training for the cifar images.
print("**************** Training on CIFAR-100 *****************")
model = Model()
model.train(x_train_cifar,y_train_cifar)
_ = model.test(x_test_cifar,y_test_cifar)

# To empty the RAM
del x_train_cifar
del y_train_cifar
del x_test_cifar
del y_test_cifar
del model

time.sleep(5)

# Could not train on this dataset due to the big data size.
# The dataset takes up to 7Go de RAM.
print("Loading PA100K ...")
x_train_PA100k, y_train_PA100k, x_test_PA100k, y_test_PA100k = load_PA100K()
print("Done")
print(len(x_train_PA100k))

# Training for the PA100K images.
print("**************** Training on PA100K *****************")
model = Model()
model.train(x_train_PA100k,y_train_PA100k)
_ = model.test(x_test_PA100k,y_test_PA100k)

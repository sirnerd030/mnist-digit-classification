import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Part (a)
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print the number of images in each set and the image dimensions
print(f'Number of training images: {x_train.shape[0]}')
print(f'Number of testing images: {x_test.shape[0]}')
print(f'Image dimensions: {x_train.shape[1]} x {x_train.shape[2]}')

# Part (b)
def plot_digits(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        digit_images = images[labels == i]
        if len(digit_images) > 0:
            image = digit_images[0]
            plt.subplot(1, 10, i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(f'Digit: {i}')
            plt.axis('off')
        else:
            plt.subplot(1, 10, i + 1)
            plt.text(0.5, 0.5, f'No {i}', horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
    plt.show()

# Part (c)
print("Plotting training set digits...")
plot_digits(x_train, y_train)

# Part (d)
x_train_01 = x_train[(y_train == 0) | (y_train == 8)]
y_train_01 = y_train[(y_train == 0) | (y_train == 8)]
x_test_01 = x_test[(y_test == 0) | (y_test == 8)]
y_test_01 = y_test[(y_test == 0) | (y_test == 8)]

print(f'Selected training images for digits 0 and 8: {len(x_train_01)}')
print(f'Selected testing images for digits 0 and 8: {len(x_test_01)}')

# Part (e)
# Ensure reproducibility
np.random.seed(0)

# Create a random selection for validation
indices = np.random.choice(len(x_train_01), 500, replace=False)
x_valid_01 = x_train_01[indices]
y_valid_01 = y_train_01[indices]

# Remaining training data
remaining_indices = np.setdiff1d(np.arange(len(x_train_01)), indices)
x_train_01 = x_train_01[remaining_indices]
y_train_01 = y_train_01[remaining_indices]

# Part (f)
print(f'Number of training images after split: {len(x_train_01)}')
print(f'Number of validation images: {len(x_valid_01)}')
print(f'Number of testing images: {len(x_test_01)}')

# Part (g)
print("Plotting validation set digits...")
plot_digits(x_valid_01, y_valid_01)

# Part (h)
def calculate_center_mean(images):
    center_mean = []
    for img in images:
        center = img[12:16, 12:16]
        center_mean.append(np.mean(center))
    return np.array(center_mean)

# Calculate attributes
attr_train_01 = calculate_center_mean(x_train_01)
attr_valid_01 = calculate_center_mean(x_valid_01)
attr_test_01 = calculate_center_mean(x_test_01)

# Part (i)
plt.figure(figsize=(10, 5))
plt.scatter(np.arange(500), attr_valid_01, c=['red' if label == 0 else 'blue' for label in y_valid_01], marker='o')
plt.xlabel('Image Number')
plt.ylabel('Center 4x4 Mean Pixel Value')
plt.title('Validation Set Center 4x4 Mean Pixel Value')
plt.legend(['Digit 0', 'Digit 8'])
plt.show()

# Visualize the distribution of the attribute values
plt.figure(figsize=(10, 5))
zero_attr = attr_valid_01[y_valid_01 == 0]
eight_attr = attr_valid_01[y_valid_01 == 8]

plt.hist(zero_attr, bins=30, alpha=0.5, label='Digit 0')
plt.hist(eight_attr, bins=30, alpha=0.5, label='Digit 8')
plt.xlabel('Center 4x4 Mean Pixel Value')
plt.ylabel('Frequency')
plt.title('Distribution of Center 4x4 Mean Pixel Value for Digits 0 and 8')
plt.legend(loc='upper right')
plt.axvline(x=20, color='r', linestyle='--', label='Threshold 20')
plt.axvline(x=30, color='g', linestyle='--', label='Threshold 30')
plt.axvline(x=50, color='b', linestyle='--', label='Threshold 50')
plt.show()

# Part (j)
# Manually inspect the histogram and set a better threshold
# Trying thresholds 20, 30, and 50
thresholds = [20, 30, 50]

# Part (k)
def calculate_accuracy(attr_values, labels, threshold):
    predictions = (attr_values < threshold).astype(int) * 8
    return np.mean(predictions == labels)

# Calculate and print accuracies for different thresholds
for threshold in thresholds:
    train_accuracy = calculate_accuracy(attr_train_01, y_train_01, threshold)
    valid_accuracy = calculate_accuracy(attr_valid_01, y_valid_01, threshold)
    test_accuracy = calculate_accuracy(attr_test_01, y_test_01, threshold)

    print(f'Threshold: {threshold}')
    print(f'Training accuracy: {train_accuracy:.2f}')
    print(f'Validation accuracy: {valid_accuracy:.2f}')
    print(f'Testing accuracy: {test_accuracy:.2f}')

# Logistic Regression Approach

# Flatten the images for logistic regression
x_train_flattened = x_train_01.reshape(x_train_01.shape[0], -1)
x_valid_flattened = x_valid_01.reshape(x_valid_01.shape[0], -1)
x_test_flattened = x_test_01.reshape(x_test_01.shape[0], -1)

# Initialize and train logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train_flattened, y_train_01)

# Predict and calculate accuracies
train_preds = log_reg.predict(x_train_flattened)
valid_preds = log_reg.predict(x_valid_flattened)
test_preds = log_reg.predict(x_test_flattened)

train_accuracy = accuracy_score(y_train_01, train_preds)
valid_accuracy = accuracy_score(y_valid_01, valid_preds)
test_accuracy = accuracy_score(y_test_01, test_preds)

print(f'Training accuracy with logistic regression: {train_accuracy:.2f}')
print(f'Validation accuracy with logistic regression: {valid_accuracy:.2f}')
print(f'Testing accuracy with logistic regression: {test_accuracy:.2f}')

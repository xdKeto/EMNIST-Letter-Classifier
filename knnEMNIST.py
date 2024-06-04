import numpy as np
from scipy.spatial import distance
from PIL import Image
import matplotlib.pyplot as plt

def write_image(image, path):
    img = Image.fromarray(np.array(image), 'L')
    img.save(path)

DATA_DIR = 'data/'
TEST_DIR = 'tes/'
TEST_DATA_FILENAME = DATA_DIR + 'emnist-letters-test-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + 'emnist-letters-test-labels-idx1-ubyte'

TRAIN_DATA_FILENAME = DATA_DIR + 'emnist-letters-train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + 'emnist-letters-train-labels-idx1-ubyte'

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

def read_images(filename, n_max_images=None):
    with open(filename, 'rb') as f:
        _ = f.read(4)  
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        images = np.frombuffer(f.read(n_images * n_rows * n_columns), dtype=np.uint8)
        images = images.reshape((n_images, n_rows, n_columns))
  
    return images

def read_labels(filename, n_max_labels=None):
    with open(filename, 'rb') as f:
        _ = f.read(4) 
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        labels = np.frombuffer(f.read(n_labels), dtype=np.uint8)
    return labels

def normalize_images(images):
    return images / 255.0

def flatten_images(images):
    return images.reshape(images.shape[0], -1)

def knn(x_train, y_train, x_test, k=3):
    y_pred = []
    for test_sample in x_test:
        distances = distance.cdist([test_sample], x_train, 'euclidean')[0]
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = [y_train[idx] for idx in nearest_indices]
        most_common_label = max(set(nearest_labels), key=nearest_labels.count)
        y_pred.append(most_common_label)
    return y_pred

def number_to_letter(n):
    return chr(n + 64) 

def rotate_and_flip(images):
    return np.array([np.fliplr(np.rot90(image, k=3)) for image in images])

def main():
    n_train = 10000
    n_test = 10000

    x_train = read_images(TRAIN_DATA_FILENAME, n_train)
    y_train = read_labels(TRAIN_LABELS_FILENAME, n_train)
    x_test = read_images(TEST_DATA_FILENAME, n_test)
    y_test = read_labels(TEST_LABELS_FILENAME, n_test)

    x_train = rotate_and_flip(x_train)
    x_test = rotate_and_flip(x_test)

    for idx, test_sample in enumerate(x_test[:10]):
        write_image(test_sample, f'{TEST_DIR}{idx}.png')
    x_test = np.array(x_test)

    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)

    x_train = flatten_images(x_train)
    x_test = flatten_images(x_test)

    random_indices = np.random.choice(len(x_test), 10, replace=False)
    random_x_test = x_test[random_indices]
    random_y_test = y_test[random_indices]

    y_pred = knn(x_train, y_train, random_x_test)

    y_pred_letters = [number_to_letter(y) for y in y_pred]
    random_y_test_letters = [number_to_letter(y) for y in random_y_test]

    accuracy = sum([
        int(y_pred_i == y_test_i)
        for y_pred_i, y_test_i
        in zip(y_pred, random_y_test)]) / len(random_y_test)

    print(f'Accuracy: {accuracy * 100}%')
    print(f'Predicted Labels: {y_pred_letters}')
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for idx, (image, pred, actual) in enumerate(zip(random_x_test, y_pred_letters, random_y_test_letters)):
        axes[idx].imshow(image.reshape(28, 28), cmap='gray')
        axes[idx].set_title(f'Prediction: {pred}, Label: {actual}')
        axes[idx].axis('off')
    plt.tight_layout()
    plt.show()
    
        

if __name__ == '__main__':
    main()

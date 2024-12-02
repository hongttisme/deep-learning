import numpy as np
import matplotlib.pyplot as plt
import random
import time
from PIL import Image


def round(x):
    return int(np.round(x))


def step(x):
    if x <= 0:
        return 0
    else:
        return 1


def resize_image(image, factor):
    new_image = np.zeros((int(np.round(image.shape[0] * factor)), int(np.round(image.shape[1] * factor))))
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i, j] = image[round(i / new_image.shape[0] * image.shape[0]), round(j / new_image.shape[1] * image.shape[1])]

    return new_image


def imshow_gray(image_2d):
    plt.figure(figsize=(7, 7))
    plt.imshow(image_2d, cmap="gray")
    

def histogram(samples, number_of_groups, lower_bound, upper_bound):
    number_of_samples = len(samples)
    counter_array = np.array([0] * number_of_groups)

    for i in range(number_of_samples):
        if samples[i] <= upper_bound and samples[i] >= lower_bound:
            counter_array[int(min((samples[i] - lower_bound) / (upper_bound - lower_bound) * number_of_groups, len(counter_array) - 1))] += 1

    x_axis_array = np.linspace(lower_bound, upper_bound, number_of_groups + 1)[:-1] + (upper_bound - lower_bound) / number_of_groups / 2
    plt.bar(x_axis_array, counter_array, width=0.8 * (upper_bound - lower_bound) / number_of_groups)
    plt.show()

    return counter_array


def open_image(file_path):
    image = Image.open(file_path)
    image_3d_array = np.array(image, dtype=int)

    if len(image_3d_array.shape) == 2:
        image_3d_array = np.tile(image_3d_array[:, :, np.newaxis], (1, 1, 3))

    # plt.figure(figsize=(12, 12))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image_3d_array)
    image_2d_array = np.int0(np.sum(image_3d_array, axis=2) / 3)
    # plt.subplot(1, 2, 2)
    # plt.imshow(image_2d_array, cmap="gray")

    return image_3d_array, image_2d_array


def save_image(image_array_2d):
    print("saving image...")
    image_save = Image.fromarray(image_array_2d)
    image_save = image_save.convert("RGB")
    image_save.save("D:\\javapngs\\python" + str(int(time.time())) + ".png")
    print("saved image successfully")


def convolve(image, kernel): 
    r = int(kernel.shape[0] / 2)
    padded_image = np.pad(image, r)
    new_image = np.zeros(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            m = padded_image[i:i + len(kernel), j:j + len(kernel)]
            sum = np.dot(m.flatten(), kernel.flatten())

            new_image[i, j] = sum

    return new_image


def median_filter(image, kernel_length): 
    r = int(kernel_length / 2)
    padded_image = np.pad(image, r)
    new_image = np.zeros(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            m = padded_image[i:i + kernel_length, j:j + kernel_length]
            median = np.median(m)

            new_image[i, j] = median

    return new_image


def compare_2_images(image_1, image_2):
    plt.figure(figsize=(13, 13))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(image_2, cmap="gray")
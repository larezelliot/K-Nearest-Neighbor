import csv
import numpy as np
from PIL import Image


def openCSV(filePath: str) -> list:
    """
    Opens a CSV file and reads its data.

    Args:
        filePath: The name of the .csv file.

    Returns:
        A tuple containg each entry of the .csv file.
    """
    with open(filePath, 'r') as csv_file:
        csv_data = []
        for row in csv.reader(csv_file):
            csv_data.append(row)
        return csv_data


i = 0


def f(x):
    global i
    w, h = 28, 28

    data = np.uint8(x).reshape(28, 28)
    img = Image.fromarray(data, 'L')
    img.save(f'./Images/Training/image{i}.png')
    i += 1


if __name__ == '__main__':
    for row in openCSV('MNIST_training_HW2.csv')[1:]:
        f(row[1:])

import csv
import time
import matplotlib.pyplot as plt


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


def distance(parameters_a: tuple, parameters_b: tuple):
    if len(parameters_a) != len(parameters_b):
        raise Exception(
            'Distance between two parameters from different lenght')

    sum = 0
    for i in range(len(parameters_a)):
        sum += (float(parameters_a[i]) - float(parameters_b[i])) ** 2
    dist = sum ** (0.5)
    return float(dist)


class K_Nearest_Neighboor:

    def __init__(self, trainingFilePath: str):
        training_parameters = []
        training_labels = []
        for row in openCSV(trainingFilePath)[1:]:
            training_parameters.append(tuple(row[1:]))
            training_labels.append(row[0])
        self.training_data = tuple(zip(training_parameters, training_labels))

    def predict(self, parameters_new: tuple, k: int):
        dist_and_labels = []

        for el in self.training_data:
            parameters = el[0]
            label = el[1]
            dist = distance(parameters_new, parameters)
            dist_and_labels.append((dist, label))
        dist_and_labels = sorted(dist_and_labels)

        labels_neighbors = {'0': 0,
                            '1': 0,
                            '2': 0,
                            '3': 0,
                            '4': 0,
                            '5': 0,
                            '6': 0,
                            '7': 0,
                            '8': 0,
                            '9': 0
                            }
        new_label = '-1'
        new_label_appearances = 0

        for i in range(k):
            label = dist_and_labels[i][1]
            labels_neighbors[label] += 1
            if labels_neighbors[label] > new_label_appearances:
                new_label = label
                new_label_appearances = labels_neighbors[label]

        return new_label


if __name__ == '__main__':
    knn = K_Nearest_Neighboor('MNIST_training_HW2.csv')

    test_parameters = []
    test_labels = []

    for row in openCSV('MNIST_test_HW2.csv')[1:]:
        test_parameters.append(tuple(row[1:]))
        test_labels.append(row[0])
    test_data = tuple(zip(test_parameters, test_labels))

    right_predictions = 0
    wrong_predictions = 0
    start = time.time()
    i = 0
    for el in test_data:
        parameters = el[0]
        label = el[1]

        predicted_label = knn.predict(parameters, k=9)
        if predicted_label == label:
            right_predictions += 1
        else:
            print(f'Wrong prediction at Image{i}.')
            print(f'\t({label}) expected ({predicted_label}) found.\n')
            wrong_predictions += 1
        i += 1
    end = time.time()

    accuracy = right_predictions / \
        (right_predictions+wrong_predictions)*100.00

    print(f'k=9\t\tAccuracy on predictions: {accuracy}%\n')

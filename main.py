import csv
import time


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
            training_labels.append(tuple(row[0]))
        self.training_data = tuple(zip(training_parameters, training_labels))

    def predict(self, parameters_new: tuple):
        label_new = '-1'
        dist_new = float('inf')

        for el in self.training_data:
            parameters = el[0]
            label = el[1]
            dist = distance(parameters_new, parameters)
            if dist < dist_new:
                dist_new = dist
                label_new = label
        return label_new


if __name__ == '__main__':
    knn = K_Nearest_Neighboor('MNIST_training_HW2.csv')

    test_parameters = []
    test_labels = []

    for row in openCSV('MNIST_test_HW2.csv')[1:]:
        test_parameters.append(tuple(row[1:]))
        test_labels.append(tuple(row[0]))
    test_data = tuple(zip(test_parameters, test_labels))

    right_predictions = 0
    wrong_predictions = 0

    start = time.time()
    i = 0
    for el in test_data:
        parameters = el[0]
        label = el[1]

        predicted_label = knn.predict(parameters)
        if predicted_label == label:
            right_predictions += 1
        else:
            print(f'Wrong prediction at {i}.')
            print(f'\t({label}) expected ({predicted_label} found).')
            wrong_predictions += 1
        i += 1
    end = time.time()
    print(
        f'Time in seconds to analyze \'MNIST_test_HW2.csv\' using KNN on \'MNIST_training_HW2.csv\': {end-start}')
    print(
        f'Accuracy on predictions: {right_predictions/(right_predictions+wrong_predictions)*100.00}%')

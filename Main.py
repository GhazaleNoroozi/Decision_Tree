from matplotlib import pyplot as plt
from sklearn import tree
import csv
import matplotlib
import numpy as np
import pandas
import math


def prepare_data(file_path):
    col_list = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
                'Embarked']
    columns = pandas.read_csv(file_path, delimiter=',', usecols=col_list)

    passengerId = columns['PassengerId']
    pclass = columns['Pclass']
    name = columns['Name']

    sex, labels = pandas.factorize(columns['Sex'])

    age = columns['Age']
    nan = age[5]
    sum = 0
    length = 0
    for record in age:
        if not math.isnan(record):
            sum += record
            length += 1

    average = sum/length
    age = [average if math.isnan(x) else x for x in age]

    sibSp = columns['SibSp']
    parch = columns['Parch']

    ticket, labels = pandas.factorize(columns['Ticket'])

    fare = columns['Fare']

    cabin, labels = pandas.factorize(columns['Cabin'])

    embarked, labels = pandas.factorize(columns['Embarked'])

    # target = columns['Survived']
    training = np.transpose(np.array([pclass, sex, age, sibSp, parch, fare, cabin, embarked], dtype=object))

    return training


# def prepare_dataset():
#     train_file = open('train.csv')
#     csv_reader = csv.reader(train_file, delimiter=',')
#
#     survived = []
#     age = []
#
#     sum = 0
#     count = 0
#
#     line_count = -1
#     for row in csv_reader:
#         line_count += 1
#         if line_count == 0:
#             continue
#         if row[5] != '':
#             sum += float(row[5])
#             count += 1
#
#         survived.append(int (row[1]))
#         if row[5] != '':
#             age.append([float(row[5])])
#         else:
#             age.append([-1])
#
#     for a in age:
#         if a[0] == -1:
#             a[0] = sum/count
#
#     # x = np.array(survived)
#     # x.reshape(1, -1)
#     return survived, age


def main():

    # tree.export_graphviz(decision_tree)
    # dotfile.close()

    x = prepare_data('train.csv')
    columns = pandas.read_csv('train.csv', delimiter=',', usecols=['Survived'])
    y = columns['Survived']
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(x, y)
    plot = tree.plot_tree(decision_tree=decision_tree)
    # plt.show()

    x = prepare_data('test.csv')
    print(x)
    prediction = decision_tree.predict(x)



if __name__ == '__main__':
    main()

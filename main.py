# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import Bert


intent_classifier= ""
def print_hi(name):
    print(intent_classifier.get_prediction("hello, I am vraj"))

# def set_up_dataset():


if __name__ == '__main__':
    intent_classifier=Bert.BertClassification()
    print_hi("Vraj")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

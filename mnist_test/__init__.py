from mnist_test.mnist_load import MnistTest
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mtest = MnistTest()
    model = mtest.create_model()
    # i, predictions_array, true_label, img
    predictions =  model[0]
    test_labels = model[1]
    img = model[2]

    i = 5
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    mtest.plot_image(i, predictions, test_labels, img)
    plt.subplot(1, 2, 2)
    mtest.plot_value_array(i, predictions, test_labels)
    plt.show()
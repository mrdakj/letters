import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from classification_report import ClassificationReport


class Model:
    def __init__(self, img_size, class_names, model_dir, report_dir):
        self.class_names = class_names
        self.model_dir = model_dir
        self.report_dir = report_dir
        self.model = self.__create_model(img_size)
        self.history = None

    def __create_model(self, img_size):
        model = tf.keras.Sequential([ 
            tf.keras.Input(shape=(img_size,img_size,1)),
            tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'),
            tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(256,activation='relu'),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])

        model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        return model


    def train(self, train_ds, val_ds):
        MCP = ModelCheckpoint(f'{self.model_dir}/model.h5',verbose=1,save_best_only=True,monitor='val_accuracy',mode='max')
        ES = EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=0,restore_best_weights = True,patience=5,mode='max')
        RLP = ReduceLROnPlateau(monitor='val_loss',patience=5,factor=0.2,min_lr=0.0001)

        self.history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=10,
                callbacks=[MCP,ES,RLP])

    def plot_accuracy_loss(self):
        if self.history != None:
            # save graph of accuracy and loss
            epochs = self.history.epoch
            loss = self.history.history['loss']
            validation_loss = self.history.history['val_loss']

            plt.title('Gubitak')
            plt.xlabel('epoha')
            plt.ylabel('gubitak')
            plt.xticks(range(len(epochs)), range(1, len(epochs)+1))
            plt.plot(epochs, loss, c='red', label='trening')
            plt.plot(epochs, validation_loss, c='orange', label='validacija')
            plt.legend(loc='best')
            plt.savefig(f'{self.report_dir}/loss.pdf', format='pdf')
            plt.clf()

            acc = self.history.history['accuracy']
            validation_acc = self.history.history['val_accuracy']

            plt.title('Tačnost')
            plt.xlabel('epoha')
            plt.ylabel('tačnost')
            plt.xticks(range(len(epochs)), range(1, len(epochs)+1))
            plt.plot(epochs, acc, c='red', label='trening')
            plt.plot(epochs, validation_acc, c='orange', label='validacija')
            plt.legend(loc='best')
            plt.savefig(f'{self.report_dir}/accuracy.pdf', format='pdf')
            plt.clf()

    def load_best_model(self):
        self.model = tf.keras.models.load_model(f'{self.model_dir}/model.h5')

    def print_scores(self, train_ds, val_ds, test_ds):
        # evaluate the model
        train_scores = self.model.evaluate(train_ds)
        print("Train accuracy: %s: %.2f%%" % (self.model.metrics_names[1], train_scores[1]*100))
        val_scores = self.model.evaluate(val_ds)
        print("Validation accuracy: %s: %.2f%%" % (self.model.metrics_names[1], val_scores[1]*100))
        test_scores = self.model.evaluate(test_ds)
        print("Test accuracy: %s: %.2f%%" % (self.model.metrics_names[1], test_scores[1]*100))
        with open(f'{self.report_dir}/scores.txt', 'w') as f:
            f.write("Train accuracy: %s: %.2f%%\n" % (self.model.metrics_names[1], train_scores[1]*100))
            f.write("Validation accuracy: %s: %.2f%%\n" % (self.model.metrics_names[1], val_scores[1]*100))
            f.write("Test accuracy: %s: %.2f%%\n" % (self.model.metrics_names[1], test_scores[1]*100))

    def __get_true_predicted_labels(self, ds):
        val_labels = []
        val_predicted = []
        for x, y in ds:
            val_predicted.extend(self.class_names[self.model.predict(x).argmax(axis=1)])
            val_labels.extend(self.class_names[y.numpy().argmax(axis=1)])
        return val_labels, val_predicted

    def dump_report(self, val_ds, test_ds):
        report = ClassificationReport(self.class_names, self.report_dir)

        val_labels, val_predicted = self.__get_true_predicted_labels(val_ds)
        report.confusion_matrix(val_labels, val_predicted, "confusion_matrix_val")
        report.classification_report(val_labels, val_predicted, "report_val")

        test_labels, test_predicted = self.__get_true_predicted_labels(test_ds)
        report.confusion_matrix(test_labels, test_predicted, "confusion_matrix_test")
        report.classification_report(test_labels, test_predicted, "report_test")



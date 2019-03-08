from clone.prepare_data import DataHandler
from clone.model_build import RecModel
from keras import backend as K

def main():
    batch_size = 16
    max_length = 24
    n_movies = 3706
    n_genres = 18
    n_hidden_units=80
    train_epochs = 20

    dataset = DataHandler(batch_size, max_length, n_movies, n_genres)
    training_set, validation_set, n_train_user, n_val_user = dataset.get_train_data()

    model_class = RecModel(batch_size, max_length, n_hidden_units,n_movies,
                        n_genres, train_epochs, n_train_user, n_val_user)

    finalmodel = model_class.build()

    model_class.modelCompile(finalmodel)


    def scheduler(epoch):
        # 每隔5个epoch，学习率减小为原来的1/2
        if epoch == 1:
            lr = K.get_value(finalmodel.optimizer.lr)
            K.set_value(finalmodel.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
        elif epoch == 2:
            lr = K.get_value(finalmodel.optimizer.lr)
            K.set_value(finalmodel.optimizer.lr, lr * 0.2)
            print("lr changed to {}".format(lr * 0.2))
        elif epoch == 120:
            lr = K.get_value(finalmodel.optimizer.lr)
            K.set_value(finalmodel.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
        return K.get_value(finalmodel.optimizer.lr)


    model_class.train(finalmodel, training_set, validation_set, scheduler)


if __name__ == '__main__':
    main()
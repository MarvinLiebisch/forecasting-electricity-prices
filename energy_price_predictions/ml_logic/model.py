from tensorflow.keras.models import load_model


def load_model(filename):
    model = load_model(f'../../../models/{filename}')
    return model


if __name__ == "__main__":
    model = load_model('gru_model.h5')
    model.summary()

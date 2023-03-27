import sys
from energy_price_predictions.ml_logic.data_import import get_git_root
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_model(filename):

    sys.setrecursionlimit(3000)
    git_root = get_git_root()
    full_path = git_root + '/models/' + filename
    print(f"Importing model {filename} from path: {full_path} ...")
    # load_model should be here after setting recursion limit
    from tensorflow.keras.models import load_model
    model = load_model(full_path)
    print(model.summary())
    return model


if __name__ == "__main__":
    model = load_model('gru_model.h5')
    model.summary()

import pickle


def load_model(path: str) -> 'BaseModel':
    with open(path, 'rb') as file:
        return pickle.load(file)

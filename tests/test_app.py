from src.models.train_model import train


def test():
    assert train() == "Hello World"
class BaseOptimizer:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run(self, **kwargs):
        raise NotImplementedError

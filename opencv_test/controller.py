from opencv_test.model import LenaModel

class OpencvController:
    def __init__(self):
        self.model = LenaModel()

    def run(self):
        self.model.execute()
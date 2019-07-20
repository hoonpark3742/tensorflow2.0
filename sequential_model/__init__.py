from sequential_model.model import SeqModel
import matplotlib.pyplot as plt

if __name__ == '__main__':
    seq = SeqModel()
    seq.create_model()
    seq.execute()
    seq.load_model()
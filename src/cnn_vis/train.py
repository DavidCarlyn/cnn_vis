from cnn_vis.models.exps import Exp001, Exp002

def train_model():
    ex = Exp002()
    ex.train()
    

if __name__ == "__main__":
    train_model()
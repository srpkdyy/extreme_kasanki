import chainer
import chainer.functions as F
import chainer.links as L




class MLP(chainer.Chain):

    def __init__(self, n_nodes=1):
        super().__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, n_nodes)
            self.fc2 = L.Linear(n_nodes, n_nodes)
            self.fc3 = L.Linear(n_nodes, 1)


    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


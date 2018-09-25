import sys
import numpy as np
import mlp

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers




def draw_modes():
    print()
    print('1: Calculation.')
    print('2: Learning.')
    print('3: Exit.')


def calc(model):
    a = np.array(input('Enter two values separated by space: ').split(), dtype=np.float32).reshape(1, 2)
    x = chainer.Variable(a)
    print('Output: {}  (Correct: {})'.format(model(x), a.sum()))


def main():
    n_nodes = int(input('Enter the number of nodes: '))
    model = mlp.MLP(n_nodes)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    modes = ['Calculation', 'Learning', 'Exit']

    while True:
        draw_modes()
        select = int(input('Enter the number of mode: '))
        selected_mode = modes[select-1]

        if (selected_mode == 'Exit'):
            sys.exit()

        if (selected_mode == 'Calculation'):
            calc(model)

        if (selected_mode == 'Learning'):
            epoch = int(input('Enter the number of epoch: '))
            digit = input('Enter the number of learning digit: ')

            for _ in range(epoch):
                a = np.random.rand(2).reshape(1, 2).astype(np.float32) * eval('1e+' + digit)
                x = chainer.Variable(a)
                x = model(x)
                t = chainer.Variable(a.sum().reshape(1, 1))

                model.zerograds()
                loss = F.squared_error(x, t)
                loss.backward()
                optimizer.update()

                print('loss: ' + str(loss))





if __name__ == '__main__':
    main()


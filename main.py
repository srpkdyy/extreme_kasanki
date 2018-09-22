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
    x = np.array(input('Enter two values separated by space: ').split(), dtype=np.float32)
    print('Output: {}  (Correct: {})'.format(model(x), a+b))


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
                model.zerograds()

                a, b = np.random.rand(2) * eval('1e+' + digit)
                x = model(a, b)
                t = a + b

                loss = F.squared_error(x, t)
                loss.backward()
                optimizer.update()

                print('loss: ' + str(loss))





if __name__ == '__main__':
    main()


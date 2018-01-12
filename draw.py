import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def loadfile(filename):
    data = open(filename,'r')
    x = []
    y = []
    for line in data:
        x_ = int(line.split(',')[0].split(' ')[1])
        y_ = float(line.split(',')[1].split(' ')[3].replace('\n', ''))
        x.append(x_)
        y.append(y_)
    return x, y


def pltdata(x, y):
    fig = plt.figure(1)
    ax = fig.subplots(1)
    ax.plot(x,y)
    xmajorLocator = MultipleLocator(4000)
    xminorLocator = MultipleLocator(100)
    ymajorLocator = MultipleLocator(0.2)
    yminorLocator = MultipleLocator(0.05)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='minor')
    plt.xlabel('Iterations/100')
    plt.ylabel('Accurrency')
    plt.title('Supervised Learning:Traing MNIST')
    plt.savefig('deep_mnist.png',dpi=100)
    plt.show()

if __name__ == "__main__":
    x, y = loadfile("Misc/deepResult.txt")
    pltdata(x,y)
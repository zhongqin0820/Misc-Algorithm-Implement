import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def loadfile(filename):
    """
    load training data file
    :param filename: resolute path correspond to this file
    :return: x: itration, y: train accurrency
    """
    data = open(filename,'r')
    x = []
    y = []
    for line in data:
        x_ = int(line.split(',')[0].split(' ')[1])
        y_ = float(line.split(',')[1].split(' ')[3].replace('\n', ''))
        x.append(x_)
        y.append(y_)
    return x, y


def pltdata(x, y, x_f, y_f):
    """
    use matplotlib.pylib to visualize data
    :param x: take care that x equals to x_f here because they are using same itrations 
    :param y: 
    :param x_f: 
    :param y_f: 
    :return: 
    """
    fig = plt.figure(1,figsize=(9,6))
    # tick margin
    xmajorLocator = MultipleLocator(4000)
    xminorLocator = MultipleLocator(100)
    ymajorLocator = MultipleLocator(0.2)
    yminorLocator = MultipleLocator(0.05)
    # left y axis
    ax = fig.subplots(1)
    ax.plot(x,y,color="blue",linestyle="-",label="MNIST")
    ax.legend(loc="lower left",shadow=True)
    ax.set_ylabel('MNIST-Accurrency')
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='minor')
    # right y axis
    ax_f = ax.twinx()
    ax_f.plot(x_f, y_f, color="red",linestyle="-",label="fashion-MNIST")
    ax_f.legend(loc="lower right",shadow=True)
    ax_f.set_ylabel("fashion-MNIST-Accurrency")
    ax_f.xaxis.set_major_locator(xmajorLocator)
    ax_f.xaxis.set_minor_locator(xminorLocator)
    ax_f.yaxis.set_major_locator(ymajorLocator)
    ax_f.yaxis.set_minor_locator(yminorLocator)
    # common x axis display
    ax.set_xlabel('Iterations/100')
    plt.title('Supervised Learning:Traing Result')
    plt.savefig('result.png',dpi=100)
    plt.show()

if __name__ == "__main__":
    x_f, y_f = loadfile("Misc/fashion_result.txt")
    x, y = loadfile("Misc/deepResult.txt")
    pltdata(x,y,x_f,y_f)
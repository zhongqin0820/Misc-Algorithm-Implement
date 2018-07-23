import time
import numpy as np

# return k neighbors index
def naive_knn(dataset, query, k):
    """
    type: dataset
    type: query
    type: k(integer)
    rtype: array
    """
    numSamples = dataset.shape[0]
    # step1: calculate Euclidean distance
    diff = np.tile(query, (numSamples, 1)) - dataset
    squaredDiff = diff**2
    # sum is performed by row
    squaredDist = np.sum(squaredDiff, axis = 1)

    # step2: sort the distance
    sortedDistIndices = np.argsort(squaredDist)
    if k > len(sortedDistIndices):
        k = len(sortedDistIndices)
    return sortedDistIndices[0:k]

# build a big graph (normalized weight matrix)
def buildGraph(Matrix, kernel_type, rbf_sigma = None, knn_num_neighbors = None):
    """
    type: Matrix
    type: kernel_type
    type: rbf_sigma
    type: knn_num_neighbors
    rtype: affinity_matrix
    """
    num_samples = Matrix.shape[0]
    affinity_matrix = np.zeros((num_samples, num_samples), np.float32)
    if kernel_type == 'rbf':
        if rbf_sigma == None:
            raise ValueError("You should input a sigma of rbf kernel")
        for i in range(num_samples):
            row_sum = 0.0
            for j in range(num_samples):
                diff = Matrix[i, :] - Matrix[j, :]

            affinity_matrix[i][j] = np.exp(sum(diff**2) / (-2.0 * rbf_sigma**2))
            row_sum += affinity_matrix[i][j]
            affinity_matrix[i][:] /= row_sum
    elif kernel_type == 'knn':
        if knn_num_neighbors == None:
            raise ValueError("You should input a k of knn kernel")
        for i in range(num_samples):
            k_neighbors = naive_knn(Matrix, Matrix[i, :], knn_num_neighbors)
            affinity_matrix[i][k_neighbors] = 1.0 / knn_num_neighbors
    else:
        raise NameError("Not support kernel type! Use knn or rbf")
    return affinity_matrix

# label propagation
def labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type='rbf', rbf_sigma=1.5, knn_num_neighbors=10, max_iter=500, tol=1e-3):
    """
    type: Mat_Label
    type: Mat_Unlabel
    type: labels
    type: kernel_type
    type: affinity_matrix
    type: knn_num_neighbors
    type: max_iter
    type: tol
    rtype: unlabel_data_labels
    """
    # initialize
    num_label_samples = Mat_Label.shape[0]
    num_unlabel_samples = Mat_Unlabel.shape[0]
    num_samples = num_label_samples + num_unlabel_samples
    labels_list = np.unique(labels)
    num_classes = len(labels_list)
    # initialize clamp_data_label
    MatX = np.vstack((Mat_Label, Mat_Unlabel)) # ?
    clamp_data_label = np.zeros((num_label_samples, num_classes),np.float32)
    for i in range(num_label_samples):
        clamp_data_label[i][labels[i]] = 1.0
    # initializa p_l = 1;p_u = 0
    label_function = np.zeros((num_samples, num_classes), np.float32)
    label_function[0: num_label_samples] = clamp_data_label
    label_function[num_label_samples:   num_samples] = -1

    # graph construction
    affinity_matrix = buildGraph(MatX, kernel_type, rbf_sigma, knn_num_neighbors)

    # start to propagation
    iter = 0
    pre_label_function = np.zeros((num_samples, num_classes), np.float32)
    changed = np.abs(pre_label_function - label_function).sum()
    while iter < max_iter and changed > tol:
        if iter % 1 == 0:
            print('--->iter %d/%d, changed: %f'%(iter, max_iter, changed))
        pre_label_function = label_function
        iter += 1
        # propagation
        label_function = np.dot(affinity_matrix, label_function)
        # clamp
        label_function[0:num_label_samples] = clamp_data_label
        # check converge
        changed = np.abs(pre_label_function - label_function).sum()

    # get terminate label of unlabled data
    unlabel_data_labels = np.zeros(num_unlabel_samples)
    for i in range(num_unlabel_samples):
        unlabel_data_labels[i] = np.argmax(label_function[i+num_label_samples])

    return unlabel_data_labels

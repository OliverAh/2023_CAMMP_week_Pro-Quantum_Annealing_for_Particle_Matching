import numpy as np
from misc import add_inverse_and_zero

def q2_1(n):  # constraints    # 定义大矩阵的维度
    N = n ** 2

    # 创建一个n*n的矩阵，所有元素都是0
    matrix = np.zeros((n, n))

    # 将对角线上的元素设置为-1
    np.fill_diagonal(matrix, -1)

    # 将上半部分的元素设置为2
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = 2

    # 创建一个N*N的矩阵，所有元素都是0
    big_matrix = np.zeros((N, N))

    # 将小矩阵放到大矩阵的对角线上
    for i in range(N // n):
        big_matrix[i * n:(i + 1) * n, i * n:(i + 1) * n] = matrix

    # print("Q2: \n", big_matrix)

    return big_matrix


def q2_2(n):  # constraints    # 定义大矩阵的维度
    # 创建n^2 * n^2的0矩阵
    size = n ** 2
    matrix = np.zeros((size, size))

    # 主对角线设为-1
    np.fill_diagonal(matrix, -1)

    # 每隔n的对角线设为2
    for i in range(size):
        for j in range(size):
            if j > i and (j - i) % n == 0:
                matrix[i, j] = 2

    # 确保只保留上三角部分，其余部分设为0
    matrix = np.triu(matrix)

    return matrix


def q1(d):  # distance
    # Flatten the list
    d_flat = [item for sublist in d for item in sublist]
    n = len(d_flat)

    # Create an n x n matrix with zeros
    matrix = np.zeros((n, n))

    # Set the diagonal elements
    for i in range(n):
        matrix[i][i] = d_flat[i] ** 2

    # Set the upper triangular elements
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] = 2 * d_flat[i] * d_flat[j]
    # print("Q1: \n", matrix)

    return matrix

def q1_diag(d):  # distance
    n = d.shape[0]
    Q = np.zeros((n ** 2, n ** 2))

    # Objective function
    for i in range(n):
        for j in range(n):
            # Q[n * i + j, n * i + j] -= 1 / (
            #             d[i, j] + 1)  # Multiply by -1 to convert the problem into a minimization problem.

            Q[n * i + j, n * i + j] += d[i, j]

    return Q

def Q(d, l=1, inverse=False, full_q=True):

    if inverse:
        d_ = add_inverse_and_zero(d, 0.00001, 60)
        l = np.amax(d_) * 1.5  # lambda
        q11 = q1(d_)
        q22 = q2_1(len(d))
        q33 = q2_2(len(d))
        # Q = q2_1(len(d)) + q2_2(len(d))
        Q = -q1(d_)/l + (l*1.1) * q2_1(len(d)) + (l*1.1) * q2_2(len(d))
        return Q

    # print(len(d))
    if full_q:
        Q = q1(d) + l*q2_1(len(d)) + l*q2_2(len(d))
    else:
        Q = q1_diag(d) + l*q2_1(len(d)) + l*q2_2(len(d))
    # Q = l*q2_1(len(d)) + l*q2_2(len(d))
    # print("Q: \n", Q)
    return Q
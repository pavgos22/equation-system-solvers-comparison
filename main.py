import matplotlib.pyplot as plt
import math
import time


def calculate_residual_norm(A, b, x_new, N):
    r = [0] * N
    for i in range(N):
        r[i] = b[i]
        for j in range(N):
            r[i] -= A[i][j] * x_new[j]
    norm = math.sqrt(sum(r[i] ** 2 for i in range(N)))
    return norm


def jacobi(A, b, res, max_iter=100):
    N = len(b)
    D = [[0] * N for _ in range(N)]
    L = [[0] * N for _ in range(N)]
    U = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i < j:
                U[i][j] = A[i][j]
            elif i > j:
                L[i][j] = A[i][j]
            else:
                D[i][j] = A[i][j]

    x = [0] * N
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(N):
            s = 0.0
            for j in range(N):
                if j != i:
                    s += (L[i][j] + U[i][j]) * x[j]
            x_new[i] = (b[i] - s) / D[i][i]

        norm = calculate_residual_norm(A, b, x_new, N)

        if norm < res:
            return x_new, k
        x = x_new.copy()
    return x, max_iter


def gauss_seidel(A, b, res, max_iter=100):
    N = len(b)
    D = [[0] * N for _ in range(N)]
    L = [[0] * N for _ in range(N)]
    U = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i < j:
                U[i][j] = A[i][j]
            elif i > j:
                L[i][j] = A[i][j]
            else:
                D[i][j] = A[i][j]

    x = [0] * N
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(N):
            s1 = 0.0
            for j in range(i):
                s1 += L[i][j] * x_new[j]
            s2 = 0.0
            for j in range(i + 1, N):
                s2 += U[i][j] * x[j]
            x_new[i] = (b[i] - s1 - s2) / D[i][i]

        norm = calculate_residual_norm(A, b, x_new, N)

        if norm < res:
            return x_new, k
        x = x_new.copy()
    return x, max_iter


def lu_decomposition(A):
    N = len(A)
    L = [[0.0] * N for _ in range(N)]
    U = [[0.0] * N for _ in range(N)]

    for i in range(N):
        L[i][i] = 1.0

        for j in range(i + 1):
            s1 = 0.0
            for k in range(j):
                s1 += U[k][i] * L[j][k]
            U[j][i] = A[j][i] - s1

        for j in range(i, N):
            s2 = 0.0
            for k in range(i):
                s2 += U[k][i] * L[j][k]
            L[j][i] = (A[j][i] - s2) / U[i][i]

    return L, U


def lu_solve(L, U, b):
    N = len(L)
    y = [0.0 for _ in range(N)]
    x = [0.0 for _ in range(N)]

    for i in range(N):
        sum_Ly = 0.0
        for j in range(i):
            sum_Ly += L[i][j] * y[j]
        y[i] = b[i] - sum_Ly

    for i in range(N - 1, -1, -1):
        sum_Ux = 0.0
        for j in range(i + 1, N):
            sum_Ux += U[i][j] * x[j]
        x[i] = (y[i] - sum_Ux) / U[i][i]

    return x


def print_matrix(A):
    for row in A:
        print(row)


def create_system(N, ex_c):
    if ex_c == 0:
        a1 = 5 + 7
    else:
        a1 = 3
    a2 = a3 = -1
    A = [[0 for _ in range(N)] for _ in range(N)]
    b = [0 for _ in range(N)]

    for i in range(N):
        b[i] = math.sin((i + 1) * (8 + 1))

        for j in range(N):
            if i == j:
                A[i][j] = a1
            elif abs(i - j) == 1:
                A[i][j] = a2
            elif abs(i - j) == 2:
                A[i][j] = a3

    return A, b


if __name__ == '__main__':

    N = 987

    ##A
    A, b = create_system(N, 0)

    ##B
    res = 1e-9

    start = time.time()
    x, k = jacobi(A, b, res)
    end = time.time()
    print("Jacobi method:")
    print("Time: " + str(end - start) + " Number of iterations: " + str(k))
    print()

    start = time.time()
    x, k = gauss_seidel(A, b, res)
    end = time.time()
    print("Gauss-Seidel method:")
    print("Time: " + str(end - start) + " Number of iterations: " + str(k))
    print()

    ##C
    A, b = create_system(N, 1)

    start = time.time()
    x, k = jacobi(A, b, res)
    end = time.time()
    print("Jacobi method:")
    print("Time: " + str(end - start) + " Number of iterations: " + str(k))
    print()

    start = time.time()
    x, k = gauss_seidel(A, b, res)
    end = time.time()
    print("Gauss-Seidel method:")
    print("Time: " + str(end - start) + " Number of iterations: " + str(k))
    print()

    ##D
    L, U = lu_decomposition(A)
    x = lu_solve(L, U, b)

    norm = calculate_residual_norm(A, b, x, N)

    print("LU method:")
    print("Residual norm: " + str(norm))

    ##E
    N_values = [100, 500, 1000, 2000, 3000]
    jacobi_times = []
    gauss_seidel_times = []
    lu_times = []

    for N in N_values:
        A, b = create_system(N, 0)

        res = 1e-9

        start = time.time()
        x, k = jacobi(A, b, res)
        end = time.time()
        jacobi_times.append(end - start)

        start = time.time()
        x, k = gauss_seidel(A, b, res)
        end = time.time()
        gauss_seidel_times.append(end - start)

        start = time.time()
        L, U = lu_decomposition(A)
        x = lu_solve(L, U, b)
        end = time.time()
        lu_times.append(end - start)

    plt.plot(N_values, jacobi_times, label='Jacobi')
    plt.plot(N_values, gauss_seidel_times, label='Gauss-Seidel')
    plt.plot(N_values, lu_times, label='LU decomposition')
    plt.xlabel('Number of unknowns (N)')
    plt.ylabel('Time (seconds)')
    plt.title('Execution time of algorithms')

    plt.legend()
    plt.savefig("TimesComparison.png")
    plt.show()

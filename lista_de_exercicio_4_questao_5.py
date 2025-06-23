import numpy as np
from scipy.linalg import qr as scipy_qr

def modified_gram_schmidt(A):
    """
    QR decomposition using the Modified Gram-Schmidt algorithm.

    Parameters:
    A : ndarray (m x n)
        Input matrix.

    Returns:
    Q : ndarray (m x n)
        Orthonormal matrix.
    R : ndarray (n x n)
        Upper triangular matrix.
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        Q[:, i] = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], Q[:, i])
            Q[:, i] -= R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(Q[:, i])
        Q[:, i] /= R[i, i]

    return Q, R


def verify_decomposition(A, Q, R, label=""):
    """
    Verifica reconstrução e ortogonalidade da decomposição QR.

    Parameters:
    A : matriz original
    Q, R : fatoração QR
    label : string opcional para identificar a saída
    """
    recon_error = np.linalg.norm(A - Q @ R)
    ortho_error = np.round(Q.T @ Q, 10)
    print(f"\n==== Verificação: {label} ====")
    print(f"Erro de reconstrução ||A - QR||: {recon_error:.2e}")
    print("Q.T @ Q (aprox. identidade):")
    print(ortho_error)


def main():
    A = np.array([
        [1,  9,  0,  5,  3,  2],
        [-6,  3,  8,  2, -8,  0],
        [3, 15, 23, 2,  1,  7],
        [3, 57, 35, 1,  7,  9],
        [3,  6, 15, 55, 5,  21],
        [33, 7,  5,  3,  5,  7]
    ], dtype=float)

    print("Matriz A:")
    print(A)

    # Gram-Schmidt modificado
    Q1, R1 = modified_gram_schmidt(A)
    print("\nQ (Gram-Schmidt modificado):")
    print(Q1)
    print("\nR (Gram-Schmidt modificado):")
    print(R1)

    # NumPy QR
    Q2, R2 = np.linalg.qr(A)
    # SciPy QR
    Q3, R3 = scipy_qr(A)

    # Verificações
    verify_decomposition(A, Q1, R1, label="Gram-Schmidt modificado")
    verify_decomposition(A, Q2, R2, label="NumPy")
    verify_decomposition(A, Q3, R3, label="SciPy")

if __name__ == "__main__":
    main()

import numpy as np
from scipy.linalg import lu

def eliminacao_gauss_pivoteamento_parcial(A):
    """
    Realiza a decomposição LU de uma matriz A com pivoteamento parcial.

    Parâmetros:
        A (np.ndarray): matriz quadrada de entrada

    Retorna:
        L (np.ndarray): matriz triangular inferior com 1s na diagonal
        U (np.ndarray): matriz triangular superior
        P (np.ndarray): matriz de permutação
        interchanges (int): número de trocas de linhas realizadas
    """
    n = A.shape[0]
    U = A.copy().astype(float)
    L = np.eye(n)
    P = np.eye(n)
    interchanges = 0

    for i in range(n - 1):
        pivotindex = i + np.argmax(np.abs(U[i:, i]))

        if pivotindex != i:
            # Troca em U
            U[[i, pivotindex], :] = U[[pivotindex, i], :]
            # Troca em P
            P[[i, pivotindex], :] = P[[pivotindex, i], :]
            # Troca em L (colunas anteriores a i)
            if i > 0:
                L[[i, pivotindex], :i] = L[[pivotindex, i], :i]
            interchanges += 1

        if U[i, i] == 0:
            raise ValueError("Matriz é singular e não pode ser fatorada")

        # Calcula multiplicadores e atualiza L e U
        multipliers = U[i+1:, i] / U[i, i]
        U[i+1:, i+1:] -= np.outer(multipliers, U[i, i+1:])
        U[i+1:, i] = 0
        L[i+1:, i] = multipliers

    return L, U, P, interchanges

def comparar_com_scipy(A):
    print("\nMatriz original A:")
    print(A)

    L_our, U_our, P_our, _ = eliminacao_gauss_pivoteamento_parcial(A)
    print("\nMinha implementação:")
    print("L:\n", L_our)
    print("U:\n", U_our)
    print("P:\n", P_our)

    print("\nVerificação PA = LU (minha implementação):")
    print(np.allclose(P_our @ A, L_our @ U_our))

    P_scipy, L_scipy, U_scipy = lu(A)
    print("\nImplementação SciPy:")
    print("P:\n", P_scipy)
    print("L:\n", L_scipy)
    print("U:\n", U_scipy)

    print("\nVerificação P_our = P_scipy.T:", np.allclose(P_our, P_scipy.T))
    print("Verificação L_our = L_scipy:", np.allclose(L_our, L_scipy))
    print("Verificação U_our = U_scipy:", np.allclose(U_our, U_scipy))

# Teste básico
if __name__ == "__main__":
    A = np.array([[2, 1, -2], [-4, 6, 3], [-4, -2, 8]], dtype=float)
    comparar_com_scipy(A)

    print("\nTeste com matriz aleatória")
    np.random.seed(42)
    B = np.random.rand(4, 4)
    comparar_com_scipy(B)

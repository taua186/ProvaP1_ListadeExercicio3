# É permitido importar o módulo numpy apenas para o uso das funções:
# numpy.array e numpy.sqrt
import numpy as np

def produto_escalar(v1, v2):
    """
    Calcula o produto escalar entre dois vetores sem usar numpy.dot.
    Os vetores devem ter a mesma dimensão.
    """
    if len(v1) != len(v2):
        raise ValueError("Os vetores devem ter a mesma dimensão para o produto escalar.")
    
    soma = 0
    for i in range(len(v1)):
        soma += v1[i] * v2[i]
    return soma

def determinante_3x3(matriz):
    """
    Calcula o determinante de uma matriz 3x3 usando a Regra de Sarrus.
    A matriz é uma lista de listas.
    """
    #  [a, b, c]
    #  [d, e, f]
    #  [g, h, i]
    a, b, c = matriz[0]
    d, e, f = matriz[1]
    g, h, i = matriz[2]
    
    # det = a(ei - fh) - b(di - fg) + c(dh - eg)
    det = (a * (e * i - f * h) -
           b * (d * i - f * g) +
           c * (d * h - e * g))
           
    return det

def volume_vetores(vetores):
    """
    Calcula o volume de um tetraedro formado por 3 vetores no espaço 
    n-dimensional e a origem.

    Args:
        vetores (list ou tuple): Uma lista/tupla contendo 3 vetores. 
                                 Cada vetor é uma lista/tupla de números.

    Returns:
        float: O volume do tetraedro.
    """
    # --- Validação dos Argumentos ---
    if not isinstance(vetores, (list, tuple)) or len(vetores) != 3:
        raise ValueError("O argumento deve ser uma lista ou tupla contendo exatamente 3 vetores.")

    try:
        # Converte para arrays numpy como permitido
        v1 = np.array(vetores[0])
        v2 = np.array(vetores[1])
        v3 = np.array(vetores[2])
    except:
        raise TypeError("Os vetores devem conter apenas elementos numéricos.")

    if not (len(v1) == len(v2) == len(v3)):
        raise ValueError("Todos os vetores devem ter a mesma dimensão.")
    
    # --- Construção da Matriz de Gram ---
    # G_ij = v_i • v_j
    gram_matrix = [
        [produto_escalar(v1, v1), produto_escalar(v1, v2), produto_escalar(v1, v3)],
        [produto_escalar(v2, v1), produto_escalar(v2, v2), produto_escalar(v2, v3)],
        [produto_escalar(v3, v1), produto_escalar(v3, v2), produto_escalar(v3, v3)]
    ]
    
    # --- Cálculo do Determinante de Gram ---
    gram_determinante = determinante_3x3(gram_matrix)
    
    # O determinante de Gram pode ser ligeiramente negativo devido a erros de 
    # ponto flutuante, mas matematicamente é >= 0.
    # Usamos max(0, ...) para evitar erros de domínio na raiz quadrada.
    if gram_determinante < 0:
        gram_determinante = 0

    # --- Cálculo do Volume Final ---
    # V = (1/6) * sqrt(det(G))
    volume = (1/6) * np.sqrt(gram_determinante)
    
    return volume

# --- Exemplos de Uso ---

# Exemplo 1: Vetores ortogonais no espaço 3D (deveria dar volume 1/6)
vetores_3d_orto = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]
vol1 = volume_vetores(vetores_3d_orto)
print(f"Exemplo 1 (Vetores Ortogonais 3D):")
print(f"Vetores: {vetores_3d_orto}")
print(f"Volume do Tetraedro: {vol1:.6f}") # 1/6 = 0.166667
print("-" * 30)


# Exemplo 2: Vetores quaisquer no espaço 3D
vetores_3d = [
    [2, 5, -1],
    [4, 1, 1],
    [0, 6, 3]
]
vol2 = volume_vetores(vetores_3d)
print(f"Exemplo 2 (Vetores Quaisquer 3D):")
print(f"Vetores: {vetores_3d}")
print(f"Volume do Tetraedro: {vol2:.6f}")
print("-" * 30)


# Exemplo 3: Vetores no espaço 4D
vetores_4d = [
    [1, 0, 2, 0],
    [0, 1, 0, 3],
    [0, 0, 1, 1]
]
vol3 = volume_vetores(vetores_4d)
print(f"Exemplo 3 (Vetores no Espaço 4D):")
print(f"Vetores: {vetores_4d}")
print(f"Volume do Tetraedro: {vol3:.6f}")
print("-" * 30)


# Exemplo 4: Vetores linearmente dependentes (volume deve ser 0)
# v3 = v1 + v2
vetores_ld = [
    [1, 2, 3],
    [4, 5, 6],
    [5, 7, 9]  # 1+4, 2+5, 3+6
]
vol4 = volume_vetores(vetores_ld)
print(f"Exemplo 4 (Vetores Linearmente Dependentes):")
print(f"Vetores: {vetores_ld}")
print(f"Volume do Tetraedro: {vol4:.6f}")
print("-" * 30)
import numpy as np

def norma_p_matriz_2por2(matriz, p):
    """
    Estima a norma-p de uma matriz 2x2.

    Parâmetros:
        matriz (numpy.array): Matriz 2x2.
        p (float): Valor da norma-p a ser calculada.

    Retorna:
        float: Estimativa da norma-p da matriz.
    """
    if matriz.shape != (2, 2):
        raise ValueError("A matriz deve ser 2x2.")

    # Função para calcular a norma-p de um vetor
    def norma_p_vetor(vetor, p):
        return sum(abs(vi) ** p for vi in vetor) ** (1 / p)

    # Função para aplicar a matriz a um vetor
    def aplicar_matriz(matriz, vetor):
        return [
            matriz[0, 0] * vetor[0] + matriz[0, 1] * vetor[1],
            matriz[1, 0] * vetor[0] + matriz[1, 1] * vetor[1]
        ]

    # Gerar pontos no círculo unitário para norma-p
    num_pontos = 1000
    angulos = np.linspace(0, 2 * np.pi, num_pontos, endpoint=False)
    vetores_unitarios = [
        [np.cos(theta), np.sin(theta)]
        for theta in angulos
    ]
    vetores_unitarios = [
        [v[0] / norma_p_vetor(v, p), v[1] / norma_p_vetor(v, p)]
        for v in vetores_unitarios
    ]

    # Calcular a norma-p de A * x para cada vetor unitário x
    normais_p = [
        norma_p_vetor(aplicar_matriz(matriz, vetor), p)
        for vetor in vetores_unitarios
    ]

    # A norma-p da matriz é o máximo dessas normas
    return max(normais_p)

# Exemplo de uso
matriz_exemplo = np.array([[1, 2], [3, 4]])
p_valor = 2
norma = norma_p_matriz_2por2(matriz_exemplo, p_valor)
print(f"A norma-{p_valor} da matriz é: {norma:.4f}")



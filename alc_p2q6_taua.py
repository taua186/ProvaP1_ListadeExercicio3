import numpy as np

def aproximacao_truncada(A, m):
    """
    Gera uma matriz A' que é uma aproximação de A, mantendo apenas os m 
    autovalores de maior valor absoluto.

    A reconstrução é baseada na decomposição espectral A = PDP⁻¹, onde a nova
    matriz A' é calculada como A' = PD'P⁻¹, com D' contendo apenas os m
    maiores autovalores e o restante zerado.

    Args:
        A (list or np.ndarray): A matriz original (n x n).
        m (int): O número de autovalores a serem mantidos.

    Returns:
        np.ndarray: A matriz aproximada A' (n x n).

    Raises:
        ValueError: Se a matriz A não for quadrada, se m for inválido,
                    ou se a matriz A possuir autovalores complexos.
    """
    # --- 1. Validação dos Argumentos ---
    
    # Converte a entrada para um array numpy para garantir a consistência
    matriz_A = np.array(A, dtype=float)
    
    # Diretriz 1: Verifica se a matriz é quadrada
    if matriz_A.shape[0] != matriz_A.shape[1]:
        raise ValueError("Erro: A matriz de entrada deve ser quadrada.")
        
    n = matriz_A.shape[0]
    
    # Diretriz 2: Verifica se m é um número válido de autovalores a selecionar
    if not (0 <= m < n):
        raise ValueError(f"Erro: O número de autovalores a manter ({m}) "
                         f"deve ser menor que a dimensão da matriz ({n}) e não negativo.")

    # --- 2. Decomposição e Verificação dos Autovalores ---
    
    # Calcula os autovalores e autovetores de A.
    autovalores, autovetores = np.linalg.eig(matriz_A)

    # Diretriz 3: VERIFICAÇÃO ADICIONADA
    # Verifica se todos os autovalores são reais.
    # Usamos np.isclose para lidar com pequenas imprecisões numéricas.
    # Se a parte imaginária de qualquer autovalor não for próxima de zero, levanta um erro.
    if np.any(~np.isclose(autovalores.imag, 0)):
        raise ValueError("Erro: A matriz de entrada possui autovalores complexos, o que não é permitido pela diretriz.")
    
    # Se passou na verificação, podemos com segurança converter para real
    autovalores = np.real(autovalores)

    # --- 3. Seleção dos m Maiores Autovalores ---

    # Calcula o valor absoluto dos autovalores para encontrar os de maior magnitude
    abs_autovalores = np.abs(autovalores)
    
    # np.argsort() retorna os índices que ordenariam o array.
    # Pegamos os últimos 'm' índices, que correspondem aos 'm' maiores valores.
    indices_maiores_m = np.argsort(abs_autovalores)[-m:]
    
    # --- 4. Construção da Nova Matriz Diagonal D' ---
    
    # Cria um array de zeros com o mesmo formato dos autovalores originais
    novos_autovalores = np.zeros_like(autovalores, dtype=float)
    
    # Preenche o novo array com os autovalores originais nas posições selecionadas
    novos_autovalores[indices_maiores_m] = autovalores[indices_maiores_m]
    
    # Cria a matriz diagonal D' a partir do vetor de novos autovalores
    D_prime = np.diag(novos_autovalores)

    # --- 5. Reconstrução da Matriz A' ---
    
    # P é a matriz de autovetores
    P = autovetores
    # P_inv é a inversa da matriz de autovetores
    P_inv = np.linalg.inv(P)
    
    # Calcula A' = P * D' * P⁻¹
    # O operador @ é usado para multiplicação de matrizes em numpy
    A_prime = P @ D_prime @ P_inv
    
    # Retorna a parte real para descartar resíduos numéricos imaginários
    return np.real(A_prime)


# --- Exemplos de Uso ---

# Matriz original 3x3 com autovalores reais
A_original = np.array([
    [1, 3, 0],
    [2, 0, 4],
    [1, 1, 5]
])

print("--- Exemplo 1: Manter os 2 maiores autovalores de uma matriz 3x3 ---")
print("Matriz Original A:\n", A_original)
m = 2
try:
    A_truncada = aproximacao_truncada(A_original, m)
    print(f"\nMatriz A' com os {m} maiores autovalores:\n", np.round(A_truncada, 4))
    
    vals_A, _ = np.linalg.eig(A_original)
    vals_A_prime, _ = np.linalg.eig(A_truncada)
    print("\nAutovalores de A (ordenados por abs):", np.sort(np.abs(np.real(vals_A))))
    print("Autovalores de A' (ordenados por abs):", np.round(np.sort(np.abs(vals_A_prime)), 4))

except ValueError as e:
    print(e)

print("\n" + "="*60 + "\n")

# --- Exemplo 2: Teste de erro com matriz com autovalores complexos ---
print("--- Exemplo 2: Teste de erro com autovalores complexos ---")
# Matriz de rotação, que classicamente tem autovalores complexos (i, -i)
A_complexa = [[0, -1], [1, 0]]
print("Matriz com autovalores complexos:\n", np.array(A_complexa))
try:
    aproximacao_truncada(A_complexa, 1)
except ValueError as e:
    print("\nResultado:", e)

print("\n" + "="*60 + "\n")


# --- Exemplo 3: Teste de erro com m >= n ---
print("--- Exemplo 3: Teste de erro com m >= n ---")
try:
    aproximacao_truncada(A_original, 3) # m=3, n=3 -> m não é menor que n
except ValueError as e:
    print("Resultado:", e)
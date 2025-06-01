import numpy as np

def crossprod(u, v):
    """
    Calculate the cross product of two 3D vectors u and v.

    Parameters:
        u (list or np.ndarray): A 3D vector.
        v (list or np.ndarray): A 3D vector.

    Returns:
        np.ndarray: The cross product u x v.
    """
    assert len(u) == 3 and len(v) == 3, "Vectors must be 3-dimensional."
    
    result = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    ]
    
    return np.array(result)

# Define vectors u and v
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Compute u x v and v x u
u_cross_v = crossprod(u, v)
v_cross_u = crossprod(v, u)

# Compute dot products
u_cross_v_dot_u = np.dot(u_cross_v, u)
v_cross_u_dot_v = np.dot(v_cross_u, v)

# Print results
print("u x v:", u_cross_v)
print("v x u:", v_cross_u)
print("(u x v) . u:", u_cross_v_dot_u)
print("(v x u) . v:", v_cross_u_dot_v)

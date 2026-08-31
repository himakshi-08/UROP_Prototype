"""
Cryptographic Key Generation Module for Biometric Template Protection.
Generates cryptographically secure random secret keys using CSPRNG.
"""

import secrets
import numpy as np

def generate_secret_key(bit_length=128):
    """
    Generates a cryptographically secure random secret key as a binary vector of uint8 (0s and 1s).
    
    Args:
        bit_length (int): Number of secret key bits (e.g., 64, 128, 256).
        
    Returns:
        tuple: (secret_bytes, secret_binary_vector)
    """
    num_bytes = bit_length // 8
    secret_bytes = secrets.token_bytes(num_bytes)
    
    # Convert bytes to binary numpy array of 0s and 1s
    secret_vector = np.unpackbits(np.frombuffer(secret_bytes, dtype=np.uint8))
    return secret_bytes, secret_vector[:bit_length]

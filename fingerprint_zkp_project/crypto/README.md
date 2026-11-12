# Ristretto255 Cryptographic Module

This module implements the modern cryptographic primitives for privacy-preserving, zero-knowledge authentication in the Fingerprint-ZKP system.

## Key Features

1. **Ristretto255 Group (RFC 9496)**:
   - Uses libsodium's native Ristretto255 implementation.
   - Provides a canonical prime-order group of order $L = 2^{252} + 27742317777372353535851937790883648493$.
   - Eliminates the cofactor $h=8$ and small-subgroup attacks present in raw Ed25519.
   - Replaces insecure 1024-bit MODP parameters with 128-bit cryptographic security.

2. **HKDF-SHA256 Two-Factor Witness Derivation**:
   - $x = \text{HKDF-SHA256}(S \parallel K, \text{salt}, \text{context}) \pmod L$.
   - Binds the recovered 32-bit biometric secret $S$ with an independent 256-bit high-entropy user credential $K$.
   - Neutralizes the $25\%$ biometric impostor collision rate: even if an impostor fingerprint matches the template, the impostor cannot compute the Schnorr witness without the victim's physical device key $K$.

3. **Interactive Sigma Protocol & Replay Protection**:
   - 3-pass interactive protocol: Commitment $T \to$ Challenge $c \to$ Response $z$.
   - Verifier checks $z \cdot B == T + c \cdot Y$.
   - Server-side session manager binds public key and commitment to a single-use session ID with TTL expiration and commitment nonce deduplication.

## File Structure

- `schnorr.py`: Ristretto255 group operations, `SchnorrProver`, and `SchnorrVerifier`.
- `key_derivation.py`: Two-factor HKDF-SHA256 witness derivation.
- `zkp_protocol.py`: `ZKPSessionManager` with anti-replay and session expiration.
- `libsodium.dll`: Bundled Windows dynamic library for self-contained, out-of-the-box execution.


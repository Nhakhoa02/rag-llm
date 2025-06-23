"""
Encryption utilities for secure data handling.
"""

import base64
import hashlib
import os
from typing import Optional, Union, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

from .logging import get_logger


class EncryptionManager:
    """Manager for data encryption and decryption operations."""
    
    def __init__(self, key: Optional[str] = None):
        """
        Initialize encryption manager.
        
        Args:
            key: Encryption key (if not provided, will generate one)
        """
        self.logger = get_logger(__name__)
        
        if key:
            self.key = self._derive_key(key)
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
        
        # Generate RSA key pair for asymmetric encryption
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password."""
        salt = b'rag_system_salt'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data using symmetric encryption.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Base64 encoded encrypted data
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = self.cipher.encrypt(data)
            return base64.b64encode(encrypted_data).decode('utf-8')
        
        except Exception as e:
            self.logger.error("Encryption failed", error=str(e))
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt data using symmetric encryption.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted data as string
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.cipher.decrypt(encrypted_bytes)
            return decrypted_data.decode('utf-8')
        
        except Exception as e:
            self.logger.error("Decryption failed", error=str(e))
            raise
    
    def encrypt_asymmetric(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data using asymmetric encryption (RSA).
        
        Args:
            data: Data to encrypt
            
        Returns:
            Base64 encoded encrypted data
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted_data = self.public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return base64.b64encode(encrypted_data).decode('utf-8')
        
        except Exception as e:
            self.logger.error("Asymmetric encryption failed", error=str(e))
            raise
    
    def decrypt_asymmetric(self, encrypted_data: str) -> str:
        """
        Decrypt data using asymmetric encryption (RSA).
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted data as string
        """
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted_data.decode('utf-8')
        
        except Exception as e:
            self.logger.error("Asymmetric decryption failed", error=str(e))
            raise
    
    def hash_data(self, data: Union[str, bytes], algorithm: str = "sha256") -> str:
        """
        Hash data using specified algorithm.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm (md5, sha1, sha256, sha512)
            
        Returns:
            Hexadecimal hash string
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            hash_func = getattr(hashlib, algorithm)
            return hash_func(data).hexdigest()
        
        except Exception as e:
            self.logger.error("Hashing failed", error=str(e))
            raise
    
    def generate_salt(self, length: int = 32) -> str:
        """
        Generate random salt for password hashing.
        
        Args:
            length: Salt length in bytes
            
        Returns:
            Base64 encoded salt
        """
        salt = os.urandom(length)
        return base64.b64encode(salt).decode('utf-8')
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Dict[str, str]:
        """
        Hash password with salt.
        
        Args:
            password: Password to hash
            salt: Salt (if not provided, will generate one)
            
        Returns:
            Dictionary with hash and salt
        """
        if not salt:
            salt = self.generate_salt()
        
        # Combine password and salt
        salted_password = password + salt
        password_hash = self.hash_data(salted_password, "sha256")
        
        return {
            "hash": password_hash,
            "salt": salt
        }
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """
        Verify password against stored hash.
        
        Args:
            password: Password to verify
            stored_hash: Stored password hash
            salt: Salt used for hashing
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            computed_hash = self.hash_password(password, salt)["hash"]
            return computed_hash == stored_hash
        
        except Exception as e:
            self.logger.error("Password verification failed", error=str(e))
            return False
    
    def get_public_key_pem(self) -> str:
        """Get public key in PEM format."""
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')
    
    def get_private_key_pem(self) -> str:
        """Get private key in PEM format."""
        pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        return pem.decode('utf-8')
    
    def encrypt_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Encrypt metadata dictionary.
        
        Args:
            metadata: Metadata dictionary to encrypt
            
        Returns:
            Encrypted metadata string
        """
        import json
        metadata_str = json.dumps(metadata, sort_keys=True)
        return self.encrypt_data(metadata_str)
    
    def decrypt_metadata(self, encrypted_metadata: str) -> Dict[str, Any]:
        """
        Decrypt metadata dictionary.
        
        Args:
            encrypted_metadata: Encrypted metadata string
            
        Returns:
            Decrypted metadata dictionary
        """
        import json
        metadata_str = self.decrypt_data(encrypted_metadata)
        return json.loads(metadata_str)


# Global encryption manager instance
encryption_manager = EncryptionManager() 
"""Unified keyboard tokenizer interface for multiple languages."""

from typing import List, Dict, Any
from abc import ABC, abstractmethod
import string

# Import language-specific alphabets
from ns_tokenizers import (
    ALL_CYRILLIC_LETTERS_ALPHABET_ORD,
    ALL_ENGLISH_LETTERS_ALPHABET_ORD
)


class BaseKeyboardTokenizer(ABC):
    """Abstract base class for keyboard tokenizers."""
    
    @abstractmethod
    def get_token(self, char: str) -> int:
        """Get token ID for a character."""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get the total vocabulary size including special tokens."""
        pass
    
    @abstractmethod
    def get_special_tokens(self) -> List[str]:
        """Get list of special tokens."""
        pass
    
    @abstractmethod
    def get_alphabet(self) -> List[str]:
        """Get the alphabet for this keyboard."""
        pass


class EnglishKeyboardTokenizer(BaseKeyboardTokenizer):
    """English QWERTY keyboard tokenizer."""
    
    def __init__(self):
        self.alphabet = ALL_ENGLISH_LETTERS_ALPHABET_ORD  # 26 letters
        self.special_tokens = ['<unk>', '<pad>']
        
        # Build token mappings
        self.i2t = self.alphabet + self.special_tokens
        self.t2i = {t: i for i, t in enumerate(self.i2t)}
    
    def get_token(self, char: str) -> int:
        """Get token ID for a character."""
        return self.t2i.get(char.lower(), self.t2i['<unk>'])
    
    def get_vocab_size(self) -> int:
        """Get the total vocabulary size."""
        return len(self.i2t)  # 28 (26 letters + 2 special)
    
    def get_special_tokens(self) -> List[str]:
        """Get list of special tokens."""
        return self.special_tokens
    
    def get_alphabet(self) -> List[str]:
        """Get the English alphabet."""
        return self.alphabet


class CyrillicKeyboardTokenizer(BaseKeyboardTokenizer):
    """Cyrillic keyboard tokenizer."""
    
    def __init__(self):
        self.alphabet = ALL_CYRILLIC_LETTERS_ALPHABET_ORD  # 33 letters
        self.special_tokens = ['-', '<unk>', '<pad>']
        
        # Build token mappings
        self.i2t = self.alphabet + self.special_tokens
        self.t2i = {t: i for i, t in enumerate(self.i2t)}
    
    def get_token(self, char: str) -> int:
        """Get token ID for a character."""
        return self.t2i.get(char, self.t2i['<unk>'])
    
    def get_vocab_size(self) -> int:
        """Get the total vocabulary size."""
        return len(self.i2t)  # 36 (33 letters + 3 special)
    
    def get_special_tokens(self) -> List[str]:
        """Get list of special tokens."""
        return self.special_tokens
    
    def get_alphabet(self) -> List[str]:
        """Get the Cyrillic alphabet."""
        return self.alphabet


# Keyboard configuration registry
KEYBOARD_CONFIGS: Dict[str, Dict[str, Any]] = {
    'qwerty_english': {
        'tokenizer_class': EnglishKeyboardTokenizer,
        'n_keys': 29,  # 26 letters + 2 special + 1 for distance dimension
        'layout_type': 'latin'
    },
    'cyrillic': {
        'tokenizer_class': CyrillicKeyboardTokenizer,
        'n_keys': 37,  # 33 letters + 3 special + 1 for distance dimension
        'layout_type': 'cyrillic'
    },
    'default': {  # Fallback to Cyrillic for backward compatibility
        'tokenizer_class': CyrillicKeyboardTokenizer,
        'n_keys': 37,
        'layout_type': 'cyrillic'
    }
}


def create_keyboard_tokenizer(grid_name: str) -> BaseKeyboardTokenizer:
    """
    Factory function to create appropriate keyboard tokenizer.
    
    Args:
        grid_name: Name of the keyboard grid (e.g., 'qwerty_english')
    
    Returns:
        Instance of appropriate keyboard tokenizer
    """
    # Check for English keyboard
    if 'english' in grid_name.lower():
        config = KEYBOARD_CONFIGS['qwerty_english']
    elif 'cyrillic' in grid_name.lower():
        config = KEYBOARD_CONFIGS['cyrillic']
    else:
        # Default fallback
        config = KEYBOARD_CONFIGS['default']
    
    return config['tokenizer_class']()


def get_keyboard_config(grid_name: str) -> Dict[str, Any]:
    """
    Get keyboard configuration for a given grid name.
    
    Args:
        grid_name: Name of the keyboard grid
    
    Returns:
        Dictionary with keyboard configuration
    """
    if 'english' in grid_name.lower():
        return KEYBOARD_CONFIGS['qwerty_english']
    elif 'cyrillic' in grid_name.lower():
        return KEYBOARD_CONFIGS['cyrillic']
    else:
        return KEYBOARD_CONFIGS['default']
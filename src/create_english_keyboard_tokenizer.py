#!/usr/bin/env python3
"""Create an English keyboard tokenizer compatible with the system."""

import string

class KeyboardTokenizerEnglish:
    """English keyboard tokenizer compatible with feature extraction."""
    
    # All lowercase English letters + some punctuation
    ENGLISH_CHARS = list(string.ascii_lowercase) + [' ', "'", ',', '.', '-']
    
    def __init__(self):
        self.i2t = self.ENGLISH_CHARS + ['<unk>', '<pad>']
        self.t2i = {t: i for i, t in enumerate(self.i2t)}
    
    def get_token(self, char):
        """Get token ID for a character."""
        return self.t2i.get(char.lower(), self.t2i['<unk>'])


if __name__ == "__main__":
    tokenizer = KeyboardTokenizerEnglish()
    print(f"Keyboard tokenizer created with {len(tokenizer.i2t)} tokens")
    print(f"Tokens: {tokenizer.i2t}")
    print(f"\nTest encoding:")
    test_words = ["hello", "world", "test", "it's"]
    for word in test_words:
        tokens = [tokenizer.get_token(c) for c in word]
        print(f"  {word} -> {tokens}")
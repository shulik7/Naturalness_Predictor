from transformers import AutoTokenizer

def get_tokenize_func(tokenizer: AutoTokenizer):
    """
    Returns a function that tokenizes SMILES strings using a ChemBERTa-compatible tokenizer.

    The returned function expects an input dictionary `entries` with keys 'Smiles' (list of SMILES strings) and 'Is_Nature_Product' (list of labels).
    It can be used with datasets' map method and produces lists (not tensors) for compatibility with datasets.
    """

    def tokenize_func(entries):
        
        if 'Smiles' not in entries or 'Is_Nature_Product' not in entries:
            missing = [k for k in ['Smiles', 'Is_Nature_Product'] if k not in entries]
            raise KeyError(f"Missing required key(s) in entries: {', '.join(missing)}")
        
        tokens = tokenizer(
            entries['Smiles'],
            padding="max_length",
            truncation=True,
            max_length=512
        )

        tokens['labels'] = entries['Is_Nature_Product']
        return tokens

    return tokenize_func    
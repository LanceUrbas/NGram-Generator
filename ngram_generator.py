"""
N-Gram Text Generator

A from-scratch implementation of an n-gram language model that learns
patterns from a source text and generates new text in a similar style.

Fill in each function below. Docstrings describe what each one should do.
"""

import random
import re
import string
from collections import defaultdict


def load_text(filepath):
    """
    Read the contents of a text file and return it as a single string.

    Parameters
        filepath (str): path to the text file to load

    Returns
        str: the full contents of the file
    """
    with open(filepath, 'r') as f:
        return f.read()
    


def tokenize(text):
    """
    Break a string of text into a list of individual word tokens.

    Consider what counts as a token here. Do you want punctuation
    treated as its own token, stripped entirely, or attached to
    the word next to it? This choice affects how natural your
    generated text will sound.

    Parameters
        text (str): the raw text to tokenize

    Returns
        list[str]: a list of tokens in the order they appeared
    """
    tokens = []
    current = ''
    for i in range(len(text)):
        if text[i] in string.punctuation:
            if current:
                tokens.append(current) 
            tokens.append(text[i]) 
            current = ''
            continue
        elif text[i] in string.whitespace:
            if current:
                tokens.append(current)
            current = ''
            continue
        else:
            current = current + text[i]
        if i == len(text) - 1:
            tokens.append(current)
    return tokens


def build_ngram_model(tokens, n):
    """
    Build a model mapping each sequence of (n - 1) tokens to a list
    of tokens that followed that sequence somewhere in the source text.

    For example, with n = 3 and the tokens ["the", "cat", "sat", "on",
    "the", "mat"], the key ("the", "cat") should map to ["sat"], since
    "sat" is the word that followed "the cat" in the source text.

    Parameters
        tokens (list[str]): the full list of tokens from the source text
        n (int): the size of the n-gram (2 for bigrams, 3 for trigrams, etc)

    Returns
        dict: maps a tuple of (n - 1) tokens to a list of tokens that
              followed that sequence in the source text
    """
    model = defaultdict(list)
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i:i + n - 1])
        value = tokens[i + n - 1]
        model[key].append(value)
    return model


def generate_text(model, n, max_words, seed=None):
    """
    Generate new text by walking through the model, starting from a
    seed sequence and repeatedly choosing a next word at random from
    the list of words that followed that sequence in the source text.

    Parameters
        model (dict): the n-gram model built by build_ngram_model
        n (int): the n-gram size used to build the model
        num_words (int): how many words to generate
        seed (tuple[str] or None): an optional starting sequence of
              (n - 1) tokens. If None, pick a random starting key
              from the model instead.

    Returns
        str: the generated text as a single string
    """
    sequence = []
    if seed == None:
        sequence.extend(random.choice(list(model.keys())))
    else:
        starting_sequences = []
        for key in list(model.keys()):
            if key[0].lower() == seed.lower():
                starting_sequences.append(key)
        sequence.extend(random.choice(starting_sequences))
    for i in range(max_words - n + 1):
        if tuple(sequence[-(n - 1):]) in model:
            word = random.choice(model[tuple(sequence[-(n - 1):])])
        else:
            break
        sequence.append(word)

    text = sequence[0].title()
    for i in range(1, len(sequence[:max_words])):
        if sequence[i - 1] in "-/'":
            text = text + sequence[i]
        elif sequence[i] in string.punctuation:
            text = text + sequence[i]
        else:
            text = text + ' ' + sequence[i]
    return text


def main():
    filepath = "Moby_Dick.txt"
    n_lengths = [3, 4, 5]
    max_words = 200

    starter_words = [
    'The', 'I', 'It', 'He', 'She', 'They', 'We', 'You', 'A', 'An',
    'And', 'But', 'So', 'Then', 'There', 'This', 'That', 'When', 'If', 'As',
    'In', 'On', 'At', 'For', 'With', 'By', 'From', 'To', 'What', 'Where',
    'Who', 'Why', 'How', 'All', 'Some', 'Many', 'One', 'My', 'Our', 'His',
    'Her', 'Your', 'Their', 'Now', 'Just', 'Well', 'Yes', 'No', 'Or', 'Do'
    ]
    seed = random.choice(starter_words)

    text = load_text(filepath)
    print("Length of text:", len(text))

    tokens = tokenize(text)
    print("Number of tokens:", len(tokens))

    for n in n_lengths:
        model = build_ngram_model(tokens, n)
        generated = generate_text(model, n, max_words, seed)
        print(f"\n{filepath[:-4]} text generated with n = {n} and {len(model)} keys:\n" + generated)


if __name__ == "__main__":
    main()

N-Gram Text Generator

A from scratch implementation of an n-gram language model, built without relying on any external NLP libraries. It learns patterns from a source text and generates new text that mimics its style.

What it does

The program tokenizes a source text into individual words and punctuation marks, builds a model mapping sequences of words to whatever followed them in the original text, then walks through that model, picking a random valid next word at each step to generate new text.

The included example trains on the full text of Moby Dick.

Core concepts used

Sliding window modeling. The model is built by sliding a window of size n across the tokenized text, mapping each sequence of n minus one words to a list of words that followed it somewhere in the source.

Randomized text generation. Starting from either a random or user chosen seed, the generator repeatedly looks up the current sequence in the model and picks a random word from the list of valid continuations, then slides forward and repeats.

Custom tokenizing without regex. Rather than reaching for a library, tokenizing is done character by character, correctly separating words from punctuation while handling edge cases like leading punctuation, repeated whitespace, and words ending a text with no trailing punctuation.

How to run it

Run the script directly.

python3 ngram_generator.py

By default it trains on Moby_Dick.txt and prints generated text at three different values of n (3, 4, and 5), all starting from the same randomly chosen word, so the difference in output is due purely to n rather than a different starting point.

Using your own training text

Swap in any plain text file by changing the filepath in main(). A few notes on picking good source text.

Larger files produce more natural sounding output, since the model has more real patterns to draw from.

If pulling a book from a source like Project Gutenberg, strip out the header and footer license text first, otherwise that boilerplate becomes part of your model's vocabulary.

Very high values of n relative to the size of your source text will cause the model to closely reproduce the original text rather than generating anything new, since there are fewer possible continuations at each step. This is a real, first principles example of overfitting, the model has too little data relative to its context window to generalize, so it just memorizes and repeats.

Notes

While testing this project, an unusual approach helped uncover a genuine bug. Since a rare word paired with a very large n gives the model exactly one possible path forward, the generated output should exactly reproduce the source text from that point onward. Testing this revealed the generator was consistently producing output one word shorter than requested, caused by an off by one error in the word count loop. This kind of deterministic edge case turned out to be a more effective way to catch a subtle bug than looking at typical randomized output.

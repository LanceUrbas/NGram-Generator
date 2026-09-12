import unittest
from ngram_generator import tokenize, build_ngram_model, generate_text


class TestTokenize(unittest.TestCase):

    def test_single_word(self):
        self.assertEqual(tokenize("Hello"), ["Hello"])

    def test_two_words(self):
        self.assertEqual(tokenize("Hello world"), ["Hello", "world"])

    def test_trailing_punctuation(self):
        self.assertEqual(tokenize("Bye."), ["Bye", "."])

    def test_leading_punctuation(self):
        self.assertEqual(tokenize("...wait"), [".", ".", ".", "wait"])

    def test_multiple_spaces(self):
        self.assertEqual(tokenize("Hello,  world!"), ["Hello", ",", "world", "!"])

    def test_consecutive_punctuation(self):
        self.assertEqual(tokenize("Hi !!"), ["Hi", "!", "!"])

    def test_newline_between_words(self):
        self.assertEqual(tokenize("Hello\nworld"), ["Hello", "world"])

    def test_tab_between_words(self):
        self.assertEqual(tokenize("Hello\tworld"), ["Hello", "world"])

    def test_empty_string(self):
        self.assertEqual(tokenize(""), [])

    def test_only_whitespace(self):
        self.assertEqual(tokenize("   "), [])

    def test_only_punctuation(self):
        self.assertEqual(tokenize("!?."), ["!", "?", "."])

    def test_leading_and_trailing_whitespace(self):
        self.assertEqual(tokenize("  Hello world  "), ["Hello", "world"])

    def test_full_sentence(self):
        self.assertEqual(
            tokenize("This is a test."),
            ["This", "is", "a", "test", "."]
        )


class TestBuildNgramModel(unittest.TestCase):
    # Using the same example from the docstring, so you can check your
    # work against it directly.

    def setUp(self):
        self.tokens = ["the", "cat", "sat", "on", "the", "mat"]

    def test_trigram_basic_key(self):
        model = build_ngram_model(self.tokens, 3)
        self.assertIn(("the", "cat"), model)
        self.assertEqual(model[("the", "cat")], ["sat"])

    def test_bigram_repeated_key_accumulates(self):
        # "the" appears twice, once followed by "cat" and once by "mat".
        # A bigram model (n=2) should show both, since the key is just
        # ("the",) in that case.
        model = build_ngram_model(self.tokens, 2)
        self.assertIn(("the",), model)
        self.assertEqual(model[("the",)], ["cat", "mat"])

    def test_bigram_key_length(self):
        model = build_ngram_model(self.tokens, 2)
        for key in model:
            self.assertEqual(len(key), 1)

    def test_trigram_key_length(self):
        model = build_ngram_model(self.tokens, 3)
        for key in model:
            self.assertEqual(len(key), 2)

    def test_model_does_not_include_key_with_no_following_word(self):
        # The last (n - 1) tokens have nothing after them, so they
        # should never appear as a key with a missing continuation.
        model = build_ngram_model(self.tokens, 3)
        self.assertNotIn(("the", "mat"), model)

    def test_short_token_list_still_returns_dict(self):
        model = build_ngram_model(["hello"], 3)
        self.assertEqual(len(model), 0)


class TestGenerateText(unittest.TestCase):
    # generate_text relies on randomness, so instead of checking for
    # one exact output, these check that whatever gets generated
    # actually follows the rules of the model.

    def setUp(self):
        self.tokens = ["the", "cat", "sat", "on", "the", "mat"]
        self.n = 2
        self.model = build_ngram_model(self.tokens, self.n)

    def test_output_is_a_string(self):
        result = generate_text(self.model, self.n, 5)
        self.assertIsInstance(result, str)

    def test_requesting_zero_words_returns_just_the_starting_word(self):
        # With no seed given, the function always returns at least the
        # single starting word it picked, even when max_words is 0.
        # Requesting 0 words isn't a realistic use case, so this documents
        # the actual behavior rather than forcing an empty result.
        result = generate_text(self.model, self.n, 0)
        self.assertEqual(len(result.split()), 1)

    def test_every_word_in_output_came_from_source_vocabulary(self):
        # Every word generated should be a word that actually appeared
        # somewhere in the original source text, nothing invented.
        # Compared case-insensitively since the first word gets capitalized.
        vocabulary = {word.lower() for word in self.tokens}
        result = generate_text(self.model, self.n, 10)
        for word in result.split():
            self.assertIn(word.lower(), vocabulary)

    def test_consecutive_pairs_in_output_are_valid_transitions(self):
        # For a bigram model, every word should have actually followed
        # the word before it somewhere in the source text.
        result = generate_text(self.model, self.n, 15).split()
        for i in range(len(result) - 1):
            current_word = result[i]
            next_word = result[i + 1]
            key = (current_word,)
            if key in self.model:
                self.assertIn(next_word, self.model[key])

    def test_running_generate_text_twice_can_produce_different_output(self):
        # Not guaranteed on every single run given randomness, but
        # across many runs from a model with more than one valid path,
        # we should see some variation.
        results = {generate_text(self.model, self.n, 10) for _ in range(20)}
        self.assertGreater(len(results), 1)


if __name__ == "__main__":
    unittest.main()

r"""

Modified Penn Treebank Tokenizer

This is copied from http://www.nltk.org/_modules/nltk/tokenize/treebank.html, but
modified to:

# Not replace double-quotes (") with single-quotes (`` and ''). This
  fixes bugs in the `span_tokenize` implementation that ships with nltk.
# Not split on contractions (e.g., "shouldn't" does not split into "shouldn" "'t").
  This produces better tokenization when using pre-trained word vectors.
# Convert generated tokens to lowercase.
# Auto-split on non-alphanumeric characters (e.g., "close-knit" ->
  "close" "-" "knit").
"""

import re
import nltk.tokenize.api
import nltk.tokenize.util

class MacIntyreContractions:
    """
    List of contractions adapted from Robert MacIntyre's tokenizer.
    """
    CONTRACTIONS2 = [r"(?i)\b(can)(?#X)(not)\b",
                     r"(?i)\b(d)(?#X)('ye)\b",
                     r"(?i)\b(gim)(?#X)(me)\b",
                     r"(?i)\b(gon)(?#X)(na)\b",
                     r"(?i)\b(got)(?#X)(ta)\b",
                     r"(?i)\b(lem)(?#X)(me)\b",
                     r"(?i)\b(mor)(?#X)('n)\b",
                     r"(?i)\b(wan)(?#X)(na)\s"]
    CONTRACTIONS3 = [r"(?i) ('t)(?#X)(is)\b", r"(?i) ('t)(?#X)(was)\b"]
    CONTRACTIONS4 = [r"(?i)\b(whad)(dd)(ya)\b",
                     r"(?i)\b(wha)(t)(cha)\b"]

class TreebankWordTokenizer(nltk.tokenize.api.TokenizerI):
    """
    The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank.
    This is the method that is invoked by ``word_tokenize()``.  It assumes that the
    text has already been segmented into sentences, e.g. using ``sent_tokenize()``.

    This tokenizer performs the following steps:

    - split standard contractions, e.g. ``don't`` -> ``do n't`` and ``they'll`` -> ``they 'll``
    - treat most punctuation characters as separate tokens
    - split off commas and single quotes, when followed by whitespace
    - separate periods that appear at the end of line

        >>> from nltk.tokenize import TreebankWordTokenizer
        >>> s = '''Good muffins cost $3.88\\nin New York.  Please buy me\\ntwo of them.\\nThanks.'''
        >>> TreebankWordTokenizer().tokenize(s)
        ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks', '.']
        >>> s = "They'll save and invest more."
        >>> TreebankWordTokenizer().tokenize(s)
        ['They', "'ll", 'save', 'and', 'invest', 'more', '.']
        >>> s = "hi, my name can't hello,"
        >>> TreebankWordTokenizer().tokenize(s)
        ['hi', ',', 'my', 'name', 'ca', "n't", 'hello', ',']
    """

    #starting quotes
    STARTING_QUOTES = [
        (re.compile(r'(^\")'), r' \1 '),
        (re.compile(r'([ (\[{<])"'), r'\1 " '),
    ]

    #punctuation
    PUNCTUATION = [
        (re.compile(r'([:,])([^\d])'), r' \1 \2'),
        (re.compile(r'([:,])$'), r' \1 '),
        (re.compile(r'\.\.\.'), r' ... '),
        (re.compile(r'[;@#$%&]'), r' \g<0> '),
        (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '), # Handles the final period.
        (re.compile(r'[?!]'), r' \g<0> '),

        (re.compile(r"([^'])' "), r"\1 ' "),
    ]

    # Pads parentheses
    PARENS_BRACKETS = (re.compile(r'[\]\[\(\)\{\}\<\>]'), r' \g<0> ')

    # Optionally: Convert parentheses, brackets and converts them to PTB symbols.
    CONVERT_PARENTHESES = [
        (re.compile(r'\('), '-LRB-'), (re.compile(r'\)'), '-RRB-'),
        (re.compile(r'\['), '-LSB-'), (re.compile(r'\]'), '-RSB-'),
        (re.compile(r'\{'), '-LCB-'), (re.compile(r'\}'), '-RCB-')
    ]

    DOUBLE_DASHES = (re.compile(r'--'), r' -- ')

    #ending quotes
    ENDING_QUOTES = [
        (re.compile(r'"'), " \" "),
        (re.compile(r'(\S)(")'), r'\1 \2 '),
#         (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
#         (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
    ]

    # List of contractions adapted from Robert MacIntyre's tokenizer.
    _contractions = MacIntyreContractions()
    CONTRACTIONS2 = list(map(re.compile, _contractions.CONTRACTIONS2))
    CONTRACTIONS3 = list(map(re.compile, _contractions.CONTRACTIONS3))

    def tokenize(self, text, convert_parentheses=False, return_str=False):
        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text)

        # Handles parentheses.
        regexp, substitution = self.PARENS_BRACKETS
        text = regexp.sub(substitution, text)
        # Optionally convert parentheses
        if convert_parentheses:
            for regexp, substitution in self.CONVERT_PARENTHESES:
                text = regexp.sub(substitution, text)

        # Handles double dash.
        regexp, substitution = self.DOUBLE_DASHES
        text = regexp.sub(substitution, text)

        #add extra space to make things easier
        text = " " + text + " "

        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text)

        for regexp in self.CONTRACTIONS2:
            text = regexp.sub(r' \1 \2 ', text)
        for regexp in self.CONTRACTIONS3:
            text = regexp.sub(r' \1 \2 ', text)

        # We are not using CONTRACTIONS4 since
        # they are also commented out in the SED scripts
        # for regexp in self._contractions.CONTRACTIONS4:
        #     text = regexp.sub(r' \1 \2 \3 ', text)

        return text if return_str else text.split()


    def span_tokenize(self, text):
        """
        Uses the post-hoc nltk.tokens.align_tokens to return the offset spans.

            >>> from nltk.tokenize import TreebankWordTokenizer
            >>> s = '''Good muffins cost $3.88\\nin New (York).  Please (buy) me\\ntwo of them.\\n(Thanks).'''
            >>> expected = [(0, 4), (5, 12), (13, 17), (18, 19), (19, 23),
            ... (24, 26), (27, 30), (31, 32), (32, 36), (36, 37), (37, 38),
            ... (40, 46), (47, 48), (48, 51), (51, 52), (53, 55), (56, 59),
            ... (60, 62), (63, 68), (69, 70), (70, 76), (76, 77), (77, 78)]
            >>> TreebankWordTokenizer().span_tokenize(s) == expected
            True
            >>> expected = ['Good', 'muffins', 'cost', '$', '3.88', 'in',
            ... 'New', '(', 'York', ')', '.', 'Please', '(', 'buy', ')',
            ... 'me', 'two', 'of', 'them.', '(', 'Thanks', ')', '.']
            >>> [s[start:end] for start, end in TreebankWordTokenizer().span_tokenize(s)] == expected
            True

        """
        tokens = self.tokenize(text)
        return nltk.tokenize.util.align_tokens(tokens, text)

tokenizer = TreebankWordTokenizer()

def _postprocess_token(token):
    # lowercase to match fasttext
    token = token.lower()
    # fix contractions to match fasttext (shouldn't -> shouldnt)
    token = re.sub(r"(\w+)'(\w+)", r'\1\2', token)
    return token

def _subtokenize(token):
    # further split tokens on non-alphanumeric to match fasttext
    # ("close-knit" -> "close" "-" "knit")
    for t in re.split(r'([^\w0-9\']|^\'|\'$)', token):
        if len(t):
            yield t

def word_tokenize(text, postprocess = True):
    for w in tokenizer.tokenize(text):
         for t in _subtokenize(w):
            if postprocess:
                t = _postprocess_token(t)
            yield t

def span_tokenize(text):
    tokens = list(word_tokenize(text, postprocess = False))
    return nltk.tokenize.util.align_tokens(tokens, text) 
            
def full_tokenize(text):
    tokens = list(word_tokenize(text, postprocess = False))
    spans = nltk.tokenize.util.align_tokens(tokens, text)
    tokens = [_postprocess_token(t) for t in tokens]
    return zip(tokens, spans)

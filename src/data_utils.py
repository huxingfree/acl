import json
import os
import re
import string
import nltk
import tensorflow as tf
from wheel.signatures.djbec import q

from sql.SqlTemplate import SqlTemplate
from csharp.CSharpTemplate import parseCSharp
from tensorflow.python.platform import gfile


# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"\W")
_DIGIT_RE = re.compile(br"\d")


def tokenizeNL(nl, lang=None):
    nl = nl.strip().decode('utf-8').encode('ascii', 'replace')
    return re.findall(r"[\w]+|[^\s\w]", nl)


def tokenizeCode(code, lang):
    code = code.strip().decode('utf-8').encode('ascii', 'replace')
    typedCode = None
    if lang == "sql":
        query = SqlTemplate(code, regex=True)
        typedCode = query.parseSql()
    elif lang == "csharp":
        typedCode = parseCSharp(code)
    elif lang == "python":
        typedCode = q.strip().decode('utf-8').encode('ascii', 'replace').split("\\s")
    #print typedCode
    tokens = [re.sub('\s+', ' ', x.strip()) for x in typedCode]

    return tokens


def create_set(directory):
    train_nls = []
    train_codes = []
    dev_nls = []
    dev_codes = []
    test_nls = []
    test_codes = []
    f = open(directory + '/train.txt', mode='r')
    lines = f.readlines()
    for line in lines:
        _, _, nl, code, _ = line.strip().split('\t')
        train_nls.append(nl)
        train_codes.append(code)
    f.close()

    f = open(directory + '/valid.txt', mode='r')
    lines = f.readlines()
    for line in lines:
        _, _, nl, code, _ = line.strip().split('\t')
        dev_nls.append(nl)
        dev_codes.append(code)
    f.close()

    f = open(directory + '/test.txt', mode='r')
    lines = f.readlines()
    for line in lines:
        _, _, nl, code, _ = line.strip().split('\t')
        test_nls.append(nl)
        test_codes.append(code)
    f.close()

    with gfile.GFile(directory + '/train/train.nl', mode="wb") as f:
        for nl in train_nls:
            f.write(nl + b'\n')
    with gfile.GFile(directory + '/train/train.code', mode="wb") as f:
        for code in train_codes:
            f.write(code + b'\n')
    with gfile.GFile(directory + '/train/dev.nl', mode="wb") as f:
        for nl in dev_nls:
            f.write(nl + b'\n')
    with gfile.GFile(directory + '/train/dev.code', mode="wb") as f:
        for code in dev_codes:
            f.write(code + b'\n')
    with gfile.GFile(directory + '/test/test.nl', mode="wb") as f:
        for nl in test_nls:
            f.write(nl + b'\n')
    with gfile.GFile(directory + '/test/test.code', mode="wb") as f:
        for code in test_codes:
            f.write(code + b'\n')


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w.lower() for w in words if w and len(w) > 1]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False, lang=None):
    """Create vocabulary file (if it does not exist yet) from data file.

      Data file is assumed to contain one sentence per line. Each sentence is
      tokenized and digits are normalized (if normalize_digits is set).
      Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
      We write it to vocabulary_path in a one-token-per-line format, so that later
      token in the first line gets id=0, second line gets id=1, and so on.

      Args:
        vocabulary_path: path where the vocabulary will be created.
        data_path: data file that will be used to create vocabulary.
        max_vocabulary_size: limit on the size of the created vocabulary.
        tokenizer: a function to use to tokenize each data sentence;
          if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
      """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 10000 == 0:
                    print("  processing line %d" % counter)
                tokens = tf.compat.as_bytes(line)
                tokens = tokenizer(line, lang) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True, lang=None):
    """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

    if tokenizer:
        words = tokenizer(sentence, lang)
    else:
        words = basic_tokenizer(sentence)

    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True, lang=None):
    """Tokenize data file and turn into token-ids using given vocabulary file.

      This function loads data line-by-line from data_path, calls the above
      sentence_to_token_ids, and saves the result to target_path. See comment
      for sentence_to_token_ids on the details of token-ids format.

      Args:
        data_path: path to the data file in one-sentence-per-line format.
        target_path: path where the file with token-ids will be created.
        vocabulary_path: path to the vocabulary file.
        tokenizer: a function to use to tokenize each sentence;
          if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
      """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 10000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                      normalize_digits, lang=lang)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_data(data_dir, code_vocab_size, nl_vocab_size, lang, tokenizer=None):
    # Create vocabularies of the appropriate sizes.

    code_vocab_path = os.path.join(data_dir, "vocab%d.code" % code_vocab_size)
    nl_vocab_path = os.path.join(data_dir, "vocab%d.nl" % nl_vocab_size)
    create_vocabulary(code_vocab_path, data_dir + "/train/train.code", code_vocab_size, tokenizeCode, lang=lang)
    create_vocabulary(nl_vocab_path, data_dir + "/train/train.nl", nl_vocab_size, tokenizeNL)

    # Create token ids for the training data.
    nl_train_ids_path = data_dir + ("/train/train.ids%d.nl" % nl_vocab_size)
    code_train_ids_path = data_dir + ("/train/train.ids%d.code" % code_vocab_size)
    data_to_token_ids(data_dir + "/train/train.nl", nl_train_ids_path, nl_vocab_path, tokenizeNL)
    data_to_token_ids(data_dir + "/train/train.code", code_train_ids_path, code_vocab_path, tokenizeCode, lang=lang)

    # Create token ids for the development data.
    nl_dev_ids_path = data_dir + ("/train/dev.ids%d.nl" % nl_vocab_size)
    code_dev_ids_path = data_dir + ("/train/dev.ids%d.code" % code_vocab_size)
    data_to_token_ids(data_dir + "/train/dev.nl", nl_dev_ids_path, nl_vocab_path, tokenizeNL)
    data_to_token_ids(data_dir + "/train/dev.code", code_dev_ids_path, code_vocab_path, tokenizeCode, lang=lang)

    # Create token ids for the test data.
    nl_test_ids_path = data_dir + ("/test/test.ids%d.nl" % nl_vocab_size)
    code_test_ids_path = data_dir + ("/test/test.ids%d.code" % code_vocab_size)
    data_to_token_ids(data_dir + "/test/test.nl", nl_test_ids_path, nl_vocab_path, tokenizeNL)
    data_to_token_ids(data_dir + "/test/test.code", code_test_ids_path, code_vocab_path, tokenizeCode, lang=lang)
    return (code_train_ids_path, nl_train_ids_path,
            code_dev_ids_path, nl_dev_ids_path,
            code_vocab_path, nl_vocab_path)

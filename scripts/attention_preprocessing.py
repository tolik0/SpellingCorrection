import re
import nltk
from numpy.random import choice as random_choice, randint as random_randint, rand
import numpy
import copy


def load_data(file_name="D:\Programming\SpellingCorrection\data\pubmed-rct-master\PubMed_20k_RCT\\test.txt", firstn=3,
              save=True):
    """
    Load data from file and add mistakes
    :param file_name: name of file with text
    :param firstn: amount of sentences to show as example
    :param save: if True save numpy arrays with final data look at save_data
    :return: source_sentences: list of sentences with lists of characters (with mistakes)
    :return: target_sentences: list of sentences with lists of characters (correct)
    :return: vocab_to_int: dict to transform characters into int
    :return: int_to_vocab: dict to transform integers into vocab
    """
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()

    # delete all trash
    text = clean_text(text)

    # divide to sentences
    data = nltk.sent_tokenize(text)  # list of strings

    # create dicts for transforming characters to ints
    vocab_to_int, int_to_vocab = create_dicts(text)

    # calculate length of longest sentence and all text
    print("\nMax length of sentence: {}.\n".format(len(max(data, key=lambda x: len(x)))))
    start_len = sum([len(sentence) for sentence in data])

    # delete short sentences and split long
    data = [y for x in data for y in split_sentence(x)]
    data = list(filter(lambda x: len(x) < 200, data))
    data = list(filter(lambda x: len(x) > 50, data))

    # calculate final length of text and print some info about text
    finish_len = sum([len(sentence) for sentence in data])
    print("Initial length of text = {}, final length of text = {}, part of saved text = {}.".format(
        start_len, finish_len,
        finish_len / start_len))

    source_sentences = copy.deepcopy(data)
    target_sentences = copy.deepcopy(data)

    # constant which control amount of mistakes
    AMOUNT_OF_NOISE = 0.5 / len(max(data, key=lambda x: len(x)))

    # add mistakes to the text
    for i in range(len(data)):
        source_sentences[i] = add_noise_to_sentence(data[i], AMOUNT_OF_NOISE)

    # Show source and edited sentences
    print('\nFirst {} sentence:'.format(firstn))
    for i in range(firstn):
        print("\nSource --> " + "".join(source_sentences[i]))
        print("Target --> " + "".join(target_sentences[i]))
        print("Different" if "".join(source_sentences[i]) != "".join(target_sentences[i]) else "Same")

    for sentences in [source_sentences, target_sentences]:
        convert_to_numbers(sentences, 200, vocab_to_int)

    if save:
        save_data(source_sentences, target_sentences)
        return vocab_to_int, int_to_vocab
    else:
        return source_sentences, target_sentences, vocab_to_int, int_to_vocab


def clean_text(text):
    """
    Delete trash from text
    :param text: string
    :return: string
    """
    text = re.sub(r"\b(?:[a-z.]*[A-Z][a-z.]*){2,}", "", text)
    text = re.sub(r"[^a-zA-Z .]+", "", text)
    text = re.sub('\'92t', '\'t', text)
    text = re.sub('\'92s', '\'s', text)
    text = re.sub('\'92m', '\'m', text)
    text = re.sub('\'92ll', '\'ll', text)
    text = re.sub('\'91', '', text)
    text = re.sub('\'92', '', text)
    text = re.sub('\'93', '', text)
    text = re.sub('\'94', '', text)
    text = re.sub('\.', '. ', text)
    text = re.sub('\!', '! ', text)
    text = re.sub('\?', '? ', text)
    text = re.sub(' +', ' ', text)
    return text


def convert_to_numbers(data, final_length, vocab_to_int):
    """
    Add special symbols to make all sentences with same length and transform characters to integers
    :param data: list of lists of strings
    :param final_length: int, length of final sentences
    :param vocab_to_int: dict, where key is string, value - int
    """
    for i in range(len(data)):
        # add special symbols to make all sentences with same length - 200
        data[i] = list(data[i]) + ["<EOS>"] + ["<PAD>"] * (final_length - 1 - len(data[i]))
        # transform characters to ints
        data[i] = list(map(lambda x: vocab_to_int[x], data[i]))


def create_dicts(text):
    # Create a dictionary to convert the vocabulary (characters) to integers
    vocab_to_int = {}
    count = 0
    for character in text:
        if character not in vocab_to_int:
            vocab_to_int[character] = count
            count += 1

    # Add special tokens to vocab_to_int
    codes = ['<PAD>', '<EOS>']
    for code in codes:
        vocab_to_int[code] = count
        count += 1

    # Check the size of vocabulary and all of the values
    vocab_size = len(vocab_to_int)
    print("The vocabulary contains {} characters.".format(vocab_size))
    print(sorted(vocab_to_int))

    # Create another dictionary to convert integers to their respective characters
    int_to_vocab = {}
    for character, value in vocab_to_int.items():
        int_to_vocab[value] = character
    return vocab_to_int, int_to_vocab


def split_sentence(sentence):
    if len(sentence) < 200:
        return [sentence]
    for i in range(195, 0, -1):
        if sentence[i] == " ":
            break
    if i < 50:
        return [sentence]
    return [sentence[:i]] + split_sentence(sentence[i + 1:])


def add_noise_to_sentence(sentence, amount_of_noise):
    """
     Add artificial spelling mistakes to string
    :param sentence: list of words
    :param amount_of_noise: constant from 0 to 1 which show amount of mistakes
    :return: list of words with mistakes
    """

    CHARS = list("abcdefghijklmnopqrstuvwxyz")

    substitutions = {
        "a": ["a"],
        "b": ["b"],
        "c": ["c"],
        "d": ["d"],
        "e": ["e"],
        "f": ["f"],
        "g": ["g"],
        "h": ["h"],
        "i": ["i"],
        "j": ["j"],
        "k": ["k"],
        "l": ["l"],
        "m": ["m"],
        "n": ["n"],
        "o": ["o"],
        "p": ["p"],
        "q": ["q"],
        "r": ["r"],
        "s": ["s"],
        "t": ["t"],
        "u": ["u"],
        "v": ["v"],
        "w": ["w"],
        "x": ["x"],
        "y": ["y"],
        "z": ["z"],
        "A": ["A"],
        "B": ["B"],
        "C": ["C"],
        "D": ["D"],
        "E": ["E"],
        "F": ["F"],
        "G": ["G"],
        "H": ["H"],
        "I": ["I"],
        "J": ["J"],
        "K": ["K"],
        "L": ["L"],
        "M": ["M"],
        "N": ["N"],
        "O": ["O"],
        "P": ["P"],
        "Q": ["Q"],
        "R": ["R"],
        "S": ["S"],
        "T": ["T"],
        "U": ["U"],
        "V": ["V"],
        "W": ["W"],
        "X": ["X"],
        "Y": ["Y"],
        "Z": ["Z"],
        " ": [" "],
        ".": ["."]
    }

    if rand() < amount_of_noise * len(sentence):
        # Replace a character with a random character
        random_char_position = random_randint(len(sentence))
        sentence = sentence[:random_char_position] + random_choice(
            substitutions[sentence[random_char_position]]) + sentence[random_char_position + 1:]

    if rand() < amount_of_noise * len(sentence):
        # Delete a character
        random_char_position = random_randint(len(sentence))
        sentence = sentence[:random_char_position] + sentence[random_char_position + 1:]

    if rand() < amount_of_noise * len(sentence) and len(sentence) < 197:
        # Add a random character
        random_char_position = random_randint(len(sentence))
        sentence = sentence[:random_char_position] + random_choice(CHARS[:-1]) + sentence[random_char_position:]

    if rand() < amount_of_noise * len(sentence):
        # Transpose 2 characters
        random_char_position = random_randint(len(sentence) - 1)
        sentence = sentence[:random_char_position] + sentence[random_char_position + 1] + \
                   sentence[random_char_position] + sentence[random_char_position + 2:]

    return sentence


def save_data(source, target, batch_size=100):
    source = numpy.array(source)
    target = numpy.array(target)
    for i in range(source.shape[0] // batch_size):
        numpy.save(f"../data/sources/{i}", source[i * batch_size:(i + 1) * batch_size, ])
        numpy.save(f"../data/targets/{i}", target[i * batch_size:(i + 1) * batch_size, ])


def transform_data(text, vocab_to_int, final_length=200):
    """
    Transform text to list of lists of ints
    :param text: text as string
    :param vocab_to_int: dict with characters as keys and integers as values
    :return: list of lists of ints
    """

    # delete all trash
    text = clean_text(text)
    data = nltk.sent_tokenize(text)

    # delete short sentences and split long
    data = [y for x in data for y in split_sentence(x)]
    data = list(filter(lambda x: len(x) < 200, data))
    data = list(filter(lambda x: len(x) > 50, data))

    convert_to_numbers(data, final_length)

    return data


if __name__ == "__main__":
    vocab_to_int, int_to_vocab = load_data(save=True)

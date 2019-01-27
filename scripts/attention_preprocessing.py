import re
import nltk
from numpy.random import choice as random_choice, randint as random_randint, rand
import copy


def load_data(file_name="D:\Programming\SpellingCorrection\data\pubmed-rct-master\PubMed_20k_RCT\\test.txt", firstn=10):
    """
    Load data from file and add mistakes
    :param file_name: name of file with text
    :param firstn: amount of sentences to show as example
    :return: source_sentences: list of sentences with lists of characters (with mistakes)
    :return: target_sentences: list of sentences with lists of characters (correct)
    :return: vocab_to_int: dict to transform characters into int
    :return: int_to_vocab: dict to transform integers into vocab
    """
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()

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
    data = nltk.sent_tokenize(text)
    vocab_to_int, int_to_vocab = create_dicts(text)

    start_len = sum([len(sentence) for sentence in data])
    print("\nMax length of sentence: {}.\n".format(len(max(data, key=lambda x: len(x)))))

    data = list(filter(lambda x: len(x) > 50, data))
    data = [y for x in data for y in split_sentence(x)]
    data = list(filter(lambda x: len(x) < 200, data))
    finish_len = sum([len(sentence) for sentence in data])

    print("Initial length of text = {}, final length of text = {}, part of saved text = {}.".format(
        start_len, finish_len,
        finish_len / start_len))

    AMOUNT_OF_NOISE = 0.5 / len(max(data, key=lambda x: len(x)))

    source_sentences = copy.deepcopy(data)
    target_sentences = copy.deepcopy(data)

    for i in range(len(data)):
        source_sentences[i] = add_noise_to_sentence(data[i], AMOUNT_OF_NOISE)

    # Show source and edited sentences
    print('\nFirst {} sentence:'.format(firstn))
    for i in range(firstn):
        print("\nSource --> " + "".join(source_sentences[i]))
        print("Target --> " + "".join(target_sentences[i]))
        print("Different" if "".join(source_sentences[i]) != " ".join(target_sentences[i]) else "Same")

    for i in range(len(data)):
        source_sentences[i] = list(source_sentences[i]) + ["<EOS>"] + ["<PAD>"] * (199 - len(source_sentences[i]))
        target_sentences[i] = list(target_sentences[i]) + ["<EOS>"] + ["<PAD>"] * (199 - len(target_sentences[i]))
        source_sentences[i] = list(map(lambda x: vocab_to_int[x], source_sentences[i]))
        target_sentences[i] = list(map(lambda x: vocab_to_int[x], target_sentences[i]))

    return source_sentences, target_sentences, vocab_to_int, int_to_vocab


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


if __name__ == "__main__":
    source, target, vocab_to_int, int_to_vocab = load_data()
    print(source[0])

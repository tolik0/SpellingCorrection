import re
import nltk
from numpy.random import choice as random_choice, randint as random_randint, rand
import copy


def load_data(file_name="D:\Programming\SpellingCorrection\data\pubmed-rct-master\PubMed_20k_RCT\\test.txt", firstn=10):
    """
    Load data from file and add mistakes
    :param file_name: name of file with text
    :param firstn: amount of sentences to show as example
    :return: two lists of sentences with lists of words (first with mistakes, second - correct)
    """
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()

    data = nltk.sent_tokenize(re.sub(r"[^a-zA-Z .]+", " ", re.sub(r"\b(?:[a-z.]*[A-Z][a-z.]*){2,}", "", text)))
    start_len = sum([len(sentence) for sentence in data])
    print(len(max(data, key=lambda x: len(x))))

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
    print('\nFirst 10 sentence:')
    for i in range(0, 10):
        print("\nSource --> " + " ".join(source_sentences[i]))
        print("Target --> " + " ".join(target_sentences[i]))
        print("Different" if " ".join(source_sentences[i]) != " ".join(target_sentences[i]) else "Same")

    # Take a look at the initial source of target datasets
    print("\nThe source is comprised of {:,} sentences. Here are the first {}.".format(len(source_sentences), firstn))
    print("\n".join([("{}. ".format(i + 1) + " ".join(source_sentences[i])) for i in range(firstn)]))

    print("\nThe target is comprised of {:,} sentences. Here are the first {}.".format(len(target_sentences), firstn))
    print("\n".join([("{}. ".format(i + 1) + " ".join(target_sentences[i])) for i in range(firstn)]))

    return source_sentences, target_sentences


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
        "a": [],
        "b": []
    }

    if rand() < amount_of_noise * len(sentence):
        # Replace a character with a random character
        random_char_position = random_randint(len(sentence))
        sentence[random_char_position] = random_choice(substitutions[sentence[random_char_position]])

    if rand() < amount_of_noise * len(sentence):
        # Delete a character
        random_char_position = random_randint(len(sentence))
        sentence = sentence[:random_char_position] + sentence[random_char_position + 1:]

    if rand() < amount_of_noise * len(sentence):
        # Add a random character
        random_char_position = random_randint(len(sentence))
        sentence = sentence[:random_char_position] + random_choice(CHARS[:-1]) + sentence[random_char_position:]

    if rand() < amount_of_noise * len(sentence):
        # Transpose 2 characters
        random_char_position = random_randint(len(sentence) - 1)
        sentence[random_char_position] = sentence[:random_char_position] + sentence[random_char_position + 1] + \
                                         sentence[random_char_position] + sentence[random_char_position + 2:]

    return sentence


if __name__ == "__main__":
    source, target = load_data()

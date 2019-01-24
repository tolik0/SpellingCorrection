import re
import nltk
from numpy.random import choice as random_choice, randint as random_randint, rand
import copy


def add_noise_to_sentence(sentence, amount_of_noise):
    """
     Add artificial spelling mistakes to string
    :param sentence: list of words
    :param amount_of_noise: constant from 0 to 1 which show amount of mistakes
    :return: list of words with mistakes
    """

    CHARS = list("abcdefghijklmnopqrstuvwxyz")

    if rand() < amount_of_noise * len(sentence):
        # Replace a character with a random character
        random_word_position = random_randint(len(sentence))
        if len(sentence[random_word_position]):
            random_char_position = random_randint(len(sentence[random_word_position]))
            sentence[random_word_position] = sentence[random_word_position][:random_char_position] + random_choice(
                CHARS[:-1]) + sentence[random_word_position][random_char_position + 1:]

    if rand() < amount_of_noise * len(sentence):
        # Delete a character
        random_word_position = random_randint(len(sentence))
        if len(sentence[random_word_position]) > 1:
            random_char_position = random_randint(len(sentence[random_word_position]))
            sentence[random_word_position] = sentence[random_word_position][:random_char_position] + \
                                             sentence[random_word_position][random_char_position + 1:]

    if rand() < amount_of_noise * len(sentence):
        # Add a random character
        random_word_position = random_randint(len(sentence))
        if len(sentence[random_word_position]):
            random_char_position = random_randint(len(sentence[random_word_position]))
            sentence[random_word_position] = sentence[random_word_position][:random_char_position] + random_choice(
                CHARS[:-1]) + sentence[random_word_position][random_char_position:]

    if rand() < amount_of_noise * len(sentence):
        # Transpose 2 characters
        random_word_position = random_randint(len(sentence))
        if len(sentence[random_word_position]) > 1:
            random_char_position = random_randint(len(sentence[random_word_position]) - 1)
            sentence[random_word_position] = sentence[random_word_position][:random_char_position] + \
                                             sentence[random_word_position][random_char_position + 1] + \
                                             sentence[random_word_position][random_char_position] + \
                                             sentence[random_word_position][random_char_position + 2:]
    return sentence


def load_data(file_name="D:\Programming\SpellingCorrection\data\pubmed-rct-master\PubMed_20k_RCT\\train.txt"):
    """
    Load data from file and add mistakes
    :param file_name: name of file with text
    :return: two lists of sentences with lists of words (first with mistakes, second - correct)
    """
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()

    data = [nltk.word_tokenize(re.sub(r"[^a-z]+", " ", sentence.lower())) for sentence in nltk.sent_tokenize(text)]

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
    print("\nThe source is comprised of {:,} sentences. Here are the first 10.".format(len(source_sentences)))
    print("\n".join([" ".join(i) for i in source_sentences[:10]]))

    print("\nThe target is comprised of {:,} sentences. Here are the first 10.".format(len(target_sentences)))
    print("\n".join([" ".join(i) for i in target_sentences[:10]]))

    return source_sentences, target_sentences


if __name__ == "__main__":
    source_sentences, target_sentences = load_data()

import re
import nltk
from numpy.random import choice as random_choice, randint as random_randint, rand
import copy


def load_data(file_name="D:\Programming\SpellingCorrection\data\pubmed-rct-master\PubMed_20k_RCT\\test.txt", firstn=10):
    """
    Load data from file and add mistakes
    :param file_name: name of file with text
    :return: two lists of sentences with lists of words (first with mistakes, second - correct)
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
    print("\nThe source is comprised of {:,} sentences. Here are the first {}.".format(len(source_sentences), firstn))
    print("\n".join([("{}. ".format(i + 1) + " ".join(source_sentences[i])) for i in range(firstn)]))

    print("\nThe target is comprised of {:,} sentences. Here are the first {}.".format(len(target_sentences), firstn))
    print("\n".join([("{}. ".format(i + 1) + " ".join(target_sentences[i])) for i in range(firstn)]))

    return source_sentences, target_sentences


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


if __name__ == "__main__":
    source, target = load_data()

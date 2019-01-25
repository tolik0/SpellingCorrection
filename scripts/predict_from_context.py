import numpy as np


def predict_output_word(model, context_words_list, topn=10):
    """Get the probability distribution of the center word given context words.

        Parameters
        ----------
        model: gensim.models.base_any2vec.BaseWordEmbeddingsModel
            Model that predicts the output
        context_words_list : list of str
            List of context words.
        topn : int, optional
            Return `topn` words and their probabilities.

        Returns
        -------
        list of (str, float)
            `topn` length list of tuples of (word, probability).

    """
    if not model.negative:
        raise RuntimeError(
        "We have currently only implemented predict_output_word for the negative sampling scheme, "
        "so you need to have run word2vec with negative > 0 for this to work."
        )

    if not hasattr(model.wv, 'vectors') or not hasattr(model.trainables, 'syn1neg'):
        raise RuntimeError("Parameters required for predicting the output words not found.")
    
    word_vocabs = []
    for w in context_words_list:
        try:
            word_vocabs.append(model.wv.vocab[w])
        except KeyError:
            continue
         
    if not word_vocabs:
        return None

    word2_indices = [word.index for word in word_vocabs]

    l1 = np.sum(model.wv.vectors[word2_indices], axis=0)
    if word2_indices and model.cbow_mean:
        l1 /= len(word2_indices)

    # propagate hidden -> output and take softmax to get probabilities
    prob_values = np.exp(np.dot(l1, model.trainables.syn1neg.T) / 25)
    prob_values /= np.sum(prob_values)
    top_indices = np.argsort(prob_values)[-topn:][::-1]
    # returning the most probable output words with their probabilities
    return [(model.wv.index2word[index1], prob_values[index1]) for index1 in top_indices]
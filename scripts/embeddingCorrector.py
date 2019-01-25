from jellyfish import damerau_levenshtein_distance
from scripts.predict_from_context import predict_output_word
import copy


class EmbeddingCorrector:

    def __init__(self, model=None):
        """Initialize EmbeddingCorrector instance
        
        Args:
            model (gensim.models.base_any2vec.BaseWordEmbeddingsModel): gensim model with pretrained words embeddings
  
        """
        self.model = model

    def correct_sentence(self, sentence, window=2, topn=10):
        """Correct mistakes in a single sentence
        
        Args: 
            sentence (:obj:`list` of :obj:`str`): list of tokens in the sentence
            window (int): word window used to predict center word from context
            topn (int): number of most probable candidates to choose from
        Returns: 
            
        """
        sentence = copy.deepcopy(sentence)
        for i in range(len(sentence)):
            if sentence[i] not in self.model.wv.vocab:
                candidates = predict_output_word(self.model, sentence[max(0, i - window): min(len(sentence), i + window + 1)], topn=topn)
#                 candidates = self.model.wv.most_similar([sentence[i]])
                # if no candidates were found
                if candidates is None:
                    continue
                sentence[i] = min(candidates, key=lambda x: damerau_levenshtein_distance(x[0], sentence[i]))[0]
        return sentence

    def correct_text(self, text, window=2, topn=10):
        """Correct mistakes in text

        Args:
            text (:obj:`list` of :obj:`list` of :obj:`str`): list of lists of tokens in the sentence
            window (int): word window used to predict center word from context
            topn (int): number of most probable candidates to choose from
        Returns:

        """
        text = copy.deepcopy(text)
        for i in range(len(text)):
                text[i] = self.correct_sentence(text[i], window=window, topn=topn)
        return text

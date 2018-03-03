import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id

       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',..


        """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for x_key, x_value in test_set.get_all_Xlengths().items():
        output_dict = {}
        x, length = x_value    
        for m_key, m_value in models.items():
                try:
                    score = m_value.score(x, length)
                    output_dict[m_key] = score
                except:
                    output_dict[m_key] = float('-inf')
        probabilities.append(output_dict)
        guesses.append(max([(value, key) for key, value in output_dict.items()])[1])

    return (probabilities, guesses)

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
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]

    
        We are trying to predict using the models and the test_set.

        The method receives a HMM model for each word. For example, one model for the word FISH, one model for the word CHOCOLATE, so on.

        Then, we got a data set (test_set) that are X and lengths values. For each one, we want to determine which word is the most likely. 
        We are going to do that calculating the score for each data in all the models, and selecting the word corresponding to the highest 
        score model.

        guess is going to be, for each item in the test_set, the word with the highest probability model.
        and probabilities is going to have for each item in the test_set, a dictionary with the possible word and the score of the model.

        For example,

        guess[0] = "FISH"
        probabilities[0] = { "FISH": 0.9, "CHOCOLATE": 0.8, "JOHN": 0.6...}

        correspond the result of the recognizer for test_data[0] (first value X, length in test_data)

        About your last question, len(guess) = 178 and len(probabilities) = 178, but each item in probabilities is going to have a dictionary with 112 entries.

        I hope this helps


        """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    #print(models.keys())
    #print(test_set.get_all_Xlengths().keys())

    for x_key, x_value in test_set.get_all_Xlengths().items():
        output_dict = {}
        x, length = x_value    
        for m_key, m_value in models.items():
                score = m_value.score(x, length)
                try:
                    output_dict[m_key] = score
                except:
                    output_dict[m_key] = float('-inf')
        probabilities.append(output_dict)
        guesses.append(max([(value, key) for key, value in probabilities.items()])[1])

    return (probabilities, guesses)

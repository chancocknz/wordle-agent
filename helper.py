__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"
__date__ = "July 2022"


def word_to_letter_indices(word,letters):
   """ Converts a word string to list of indices from the alphabet of letters.

   :param word: a single word string
   :param letters: list of letters constituting the alphabet
   :return: a list representing the passed in word in indices of the alphabet
   """
   letter_indices = []

   for l in word:
      try:
         index = letters.index(l)
         letter_indices.append(index)
      except:
         pass

   return letter_indices


def letter_indices_to_word(letter_indices, letters):
   """ Converts a list of indices from the alphabet of letters to a word string.

   :param letter_indices: a list representing the passed in word in indices of the alphabet
   :param letters: list of letters constituting the alphabet
   :return: a single word string according to letter_indices
   """
   word = ""

   for i in letter_indices:
      word += letters[i]

   return word

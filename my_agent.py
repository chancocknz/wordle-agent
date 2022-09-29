__author__ = "Callum Hancock"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "hanca804@student.otago.ac.nz"
__date__ = "July 2022"

import helper
import numpy as np

class WordleAgent():
   """
       A class that encapsulates the code dictating the
       behaviour of the Wordle playing agent

       ...

       Attributes
       ----------
       dictionary : list
           a list of valid words for the game
       letter : list
           a list containing valid characters in the game
       word_length : int
           the number of letters per guess word
       num_guesses : int
           the max. number of guesses per game
       mode: str
           indicates whether the game is played in 'easy' or 'hard' mode

       Methods
       -------
       AgentFunction(percepts)
           Returns the next word guess given state of the game in percepts
       """

   def __init__(self, dictionary, letters, word_length, num_guesses, mode):
      """
      :param dictionary: a list of valid words for the game
      :param letters: a list containing valid characters in the game
      :param word_length: the number of letters per guess word
      :param num_guesses: the max. number of guesses per game
      :param mode: indicates whether the game is played in 'easy' or 'hard' mode
      Also adds a backup dictionary that gets reduced/rebuilt each round, a data
      field for the index of the first guess that gets determined once and reused,
      and an instance data field for the guess counter to see if that ever gets
      out of sync with the one given by the percepts.
      """

      self.dictionary = dictionary
      self.dictionary_backup = dictionary.copy()
      self.letters = letters
      self.word_length = word_length
      self.num_guesses = num_guesses
      self.mode = mode
      self.first_guess_index = 0
      self.last_guess_counter = 0


   def AgentFunction(self, percepts):
      """Returns the next word guess given state of the game in percepts

      :param percepts: a tuple of three items: guess_counter, letter_indexes, and letter_states;
               guess_counter is an integer indicating which guess this is, starting with 0 for initial guess;
               letter_indexes is a list of indexes of letters from self.letters corresponding to
                           the previous guess, a list of -1's on guess 0;
               letter_states is a list of the same length as letter_indexes, providing feedback about the
                           previous guess (conveyed through letter indexes) with values of 0 (the corresponding
                           letter was not found in the solution), -1 (the correspond letter is found in the
                           solution, but not in that spot), 1 (the corresponding letter is found in the solution
                           in that spot).
      :return: string - a word from self.dictionary that is the next guess
      """

      # Getting values from percepts
      guess_counter, letter_indexes, letter_states = percepts

      # Incrementing class-wide counter to keep track of synchronization
      self.last_guess_counter += 1

      # Calls a function that is only run once in the whole programme that
      # gets the index of the best first guess, used for every round.
      self.get_first_guess()

      # Refills the secondary dictionary if it is the first round, if not
      # it calls the reduce possible guesses method
      if guess_counter == 0:
         self.last_guess_counter = 0
         self.dictionary_backup = self.dictionary.copy()
         return self.dictionary_backup[self.first_guess_index]
      else:
         self.reduce_guesses(percepts)

      # Checks if the instance guess counter is out of sync with the one in the
      # percepts. If this is the case, we know that a guess hasn't been accepted.
      # E.g., hard mode conditions not fulfilled. So we get rid of the index of the
      # last guess.
      if guess_counter != self.last_guess_counter:
         scores = self.calculate_scores()
         max_index = np.argmax(scores)
         self.dictionary_backup.remove(self.dictionary_backup[max_index])


      # This portion of the code does the narrowing down of what the best next guess
      # will be based on the narrowed down dictionary. This is split between checking
      # if it is stuck in a situation like _RAIN or just giving scores to each word
      # left in the dictionary.
      if len(self.dictionary_backup) > 0:
         # Checks to see whether we can develop a clever guess for to avoid guessing
         # "DRAIN", "TRAIN", "BRAIN", only to find the solution is "CRAIN".
         if self.mode == 'easy' and len(self.dictionary_backup) > 2 and guess_counter != self.num_guesses - 1:
            num_greens = self.green_counter(percepts)
            if num_greens == self.word_length - 1:
               guess = self.smart_guess(percepts)
               return guess
         # If not we use our scoring algorithms that take advantage of a basic probability /
         # entropy method and return the first instance of the max score.
         scores = self.calculate_scores()
         max_index = np.argmax(scores)
         return self.dictionary_backup[max_index]

   def run_once(f):
      """
      Function that ensures helps get_first_guess literally "runs once" for the whole programme
      even though this script gets called multiple times. I am unfamiliar with wrapper
      functions and so this idea is from:
      https://stackoverflow.com/questions/4103773/efficient-way-of-having-a-function-only-execute-once-in-a-loop

      It does not help with the algorithm itself, it only helps in ensuring that the first
      guess is only determined once (this can take a few minutes), then after this it reuses
      it every time.

      :return: some function that ensures run_once only runs once.
      """
      def wrapper(*args, **kwargs):
         if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

      wrapper.has_run = False
      return wrapper

   @run_once
   def get_first_guess(self):
      """
      Determines the best possible first guess from the dictionary. Uses the
      probability / entropy functions below to score each word and picks the
      first instance of the word that scores the highest.

      :return: nothing - the index of the first word is added to the data field.
      """
      scores = self.calculate_scores()
      self.first_guess_index = np.argmax(scores)

   def smart_guess(self, percepts):
      """
      Used in situations like "_RAIN" where we are in easy mode, not on our last guess,
      have more than two words in the dictionary, and have n-1 greens where n is the
      number of letters to try to avoid wasting guesses and get the solution in only
      two more steps. E.g. by guessing DOUBT.

      Does this by adding the letters that could fill the only grey spot to an array and then
      working through the full dictionary to try and find words that include the highest number of
      these. It penalises words with double letters by 90% to encourage the most information
      possible to be found (however most often words with more than one of the letters also have
      double letters.

      :param percepts: information about the last guess
      :return: a string that represents the guess that should hopefully narrow down the possible
               solutions to one.
      """
      guess_counter, letter_indexes, letter_states = percepts
      grey_pos = letter_states.index(0)
      possible_letters = []
      scores = []
      for word in self.dictionary_backup:
         possible_letters.append(word[grey_pos])
      for i in range(len(self.dictionary)):
         scores.append(0)
         for char in self.dictionary[i]:
            double_letter = False
            if char in possible_letters:
               scores[i] += 1
               count = self.dictionary[i].count(char)
               if count > 1:
                  double_letter = True
         if double_letter:
            scores[i] *= 0.1
      smart_index = np.argmax(scores)
      return self.dictionary[smart_index]


   def green_counter(self, percepts):
      """
      Helper function to count the number of green letters in the last guess.

      :param percepts: information about the last guesses
      :return: int - for the number of green tiles in the last guess.
      """
      guess_counter, letter_indexes, letter_states = percepts
      return letter_states.count(1)

   def calculate_scores(self):
      """
      Function to break up scores function that adds each word's
      score to an array and return it. Calls the score function
      which does the heavy lifting.
      :return: - int array that is the size of the reduced dictionary
               with corresponding scores
      """
      scores = []
      for word in self.dictionary_backup:
         scores.append(self.score(word))
      return scores


   def score(self, word):
      """
      Function to calculate the entropy of each letter in a given word
      and sum them up to get the total word score. Uses a penalty of 10%
      for words with double letters.

      :param word: is the word we are calculating the score of
      :return: decimal for the score of the word
      """
      s = 0
      double_letter = False
      for i in range(len(word)):
         s += self.entropy(word[i], i)
         count = word.count(word[i])
         if count > 1:
            double_letter = True
      if double_letter:
         s *= 0.9
      return s


   def probability(self, char, pos):
      """
      Calculates the probability of a given character occuring
      in a given position by taking the frequency of occurences / N
      where N is the length of the reduced dictionary.

      :param char: the character we are scoring
      :param pos: the position we are investigating
      :return: decimal for the probability that the char occurs in that position
      """
      freq = 0
      for i in range(len(self.dictionary_backup)):
         if self.dictionary_backup[i][pos] == char:
            freq += 1
      prob = freq / len(self.dictionary_backup)
      return prob

   def entropy(self, char, pos):
      """
      Entropy function that uses the probability of a given character
      appearing at a given index and normalises this so that values closest
      to 0.5 are preferred. Idea for the formula from:
      https://dev.to/vnjogani/the-optimal-strategy-for-solving-a-wordle-5fd7

      :param char: is the car we're investigating
      :param pos: is the position of that char
      :return: a decimal for the entropy
      """
      p = self.probability(char, pos)
      e = p * (1-p)
      return e

   def reduce_guesses(self, percepts):
      """
      Reduce guesses function that takes the information from the last
      guess and uses it to reduce the size of the backup dictionary (the
      list of possible solutions). This method prioritises accuracy over
      effectiveness, meaning that we only delete words that have a grey letter
      in that very position to avoid accidentally deleting a word that shouldn't

      Extra effectiveness is applied by counting the number of instances of that
      offending grey word and if the word we are checking only has one then we
      delete it.

      The rest of the logic involves deleting words that don't have a green in the
      position of the last guess and deleting ones who have a yellow in the position
      of the last guess or do not contain the yellow letter at all.

      :param percepts: information about the last guess.
      :return: nothing - information gets added (taken away) from  the backup dictionary
      """
      guess_counter, letter_indexes, letter_states = percepts
      dict_copy = self.dictionary_backup.copy()
      in_english = helper.letter_indices_to_word(letter_indexes, self.letters)
      for j in range(len(dict_copy)):
         word = dict_copy[j]
         for i in range(len(word)):
            if letter_states[i] == 0:
               if in_english[i] in word:
                  num_instances = in_english.count(in_english[i])
                  if num_instances == 1:
                     self.dictionary_backup.remove(word)
                     break
                  else:
                     if in_english[i] == word[i]:
                        self.dictionary_backup.remove(word)
                        break
            elif letter_states[i] == 1:
               if in_english[i] != word[i]:
                  self.dictionary_backup.remove(word)
                  break
            elif letter_states[i] == -1:
               if in_english[i] not in word or in_english[i] == word[i]:
                  self.dictionary_backup.remove(word)
                  break
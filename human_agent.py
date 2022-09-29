__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"
__date__ = "July 2022"


import numpy as np
import sys
import readchar
from wordle import bcolors

class WordleAgent():
   """
             A class that encapsulates the code dictating the
             behaviour of the Wordle playing agent

             ...

             Attributes
             ----------
             word_length : int
                 the number of letters per guess word
             num_guesses : int
                 the max. number of guesses per game
             letter : list
                 a list containing valid characters in the game
             dictionary : list
                 a list of valid words for the game
             mode: str
                 indicates whether the game is played in 'easy' or 'hard' mode
            states: list
                 list of states of the letters in the alphabet for displaing the
                 map of available letters to the player

             Methods
             -------
             AgentFunction(percepts)
                 Returns the next word guess given state of the game in percepts
             """

   def __init__(self, dictionary,letters,word_length,num_guesses,mode):
      """
      :param dictionary: a list of valid words for the game
      :param letters: a list containing valid characters in the game
      :param word_length: the number of letters per guess word
      :param num_guesses: the max. number of guesses per game
      :param mode: indicates whether the game is played in 'easy' or 'hard' mode
      """

      self.dictionary = dictionary
      self.letters = letters
      self.word_length = word_length
      self.num_guesses = num_guesses
      self.mode = mode
      self.states = 2*np.ones(len(self.letters)).astype('int8')


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

      guess_counter, letter_indexes, letter_states = percepts

      if guess_counter==0:
         # On first guess, reset the states of available letters
         self.states = 2*np.ones(len(self.letters)).astype('int8')
      else:
         # Update the state of available letters based on the feedback
         # from the last guess
         for i in range(len(letter_indexes)):
            index = letter_indexes[i]
            state = letter_states[i]
            if state == 1:
               self.states[index] = 1
            elif state == -1:
               if self.states[index] != 1:
                  self.states[index] = -1
            else:
               if self.states[index] > 1:
                  self.states[index] = 0

      # If letter_states has all 1's, that means we guessed the last word
      # correctly and there is nothing else to do
      if np.sum(letter_states) == len(letter_states):
         return None

      # Print out the available letters
      sys.stdout.write("[")
      for index in range(len(self.letters)):
         state = self.states[index]

         if state == 0:
            sys.stdout.write(f"{bcolors.NOTINWORD}%c{bcolors.ENDC}" % self.letters[index])
         elif state == 1:
            sys.stdout.write(f"{bcolors.INPLACE}%c{bcolors.ENDC}" % self.letters[index])
         elif state == -1:
            sys.stdout.write(f"{bcolors.INWORD}%c{bcolors.ENDC}" % self.letters[index])
         else:
            sys.stdout.write("%c" % self.letters[index])

      sys.stdout.write("]\n\r    ")
      for i in range(len(letter_indexes)):
         sys.stdout.write("_")

      for i in range(len(letter_indexes)):
         sys.stdout.write("\b")

      sys.stdout.flush()

      # Prompt the player for a guess
      i = 0
      actions = np.zeros((len(letter_indexes))).astype('int8').tolist()
      while i < len(letter_indexes)+1:
         while True:
            c = readchar.readkey()

            if c=='\x03':
               # Ctrl-C exits the entire program
               sys.exit(-1)
            if c=='\x04':
               # Ctrl-D gives up on the current game
               sys.stdout.write("\n\r")
               for i in range(len(letter_indexes) + 4):
                  sys.stdout.write("\b")
               for i in range(len(letter_indexes) + 4):
                  sys.stdout.write("\b")
               sys.stdout.write("\r")
               return None
            elif c=='\x7f':
               # Backspace removes last character written
               if i>0:
                  sys.stdout.write("\b_\b")
                  sys.stdout.flush()
                  i -= 1
               i -= 1
               break
            elif i>0 and c>='0' and c<='9':
               letter_index = actions[i-1]+int(c)
               if letter_index < len(self.letters):
                  actions[i-1] = letter_index
                  sys.stdout.write("\b%s" % self.letters[letter_index])
                  sys.stdout.flush()
                  continue

            elif (c=='\r' or c=='\n') and i==len(letter_indexes):
               break

            c = str(c)
            c = c.upper()

            try:
               if i<len(letter_indexes):
                  letter_index = self.letters.index(c)
                  actions[i] = letter_index
                  sys.stdout.write("%s" % self.letters[letter_index])
                  sys.stdout.flush()
                  break
            except:
               pass

         i += 1

      for i in range(len(letter_indexes)+4):
         sys.stdout.write("\b")

      sys.stdout.flush()


      return actions

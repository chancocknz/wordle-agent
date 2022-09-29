__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"
__date__ = "July 2022"

import os,io,sys
import locale
from functools import cmp_to_key
import numpy as np
import importlib
import traceback
import time
from settings import game_settings

class bcolors:
   NOTINWORD = '\033[100m'
   INPLACE = '\033[102m'
   INWORD = '\033[103m'
   ENDC = '\033[0m'


def read_dictionary(file_path,word_length=None):
   with io.open(file_path, mode="r", encoding="iso-8859-15") as f:
      words = f.read()
      words = words.split("\n")
      try:
         i = words.index("<support>")
         words = [words[:i], words[i+1:]]
      except:
         words = [words]

      dictionary = []
      for i in range(len(words)):
         words[i] = map(lambda x: x.upper(), words[i])
         if word_length is not None:
            words[i] = filter(lambda x: len(x) == word_length, words[i])
         words[i] = list(filter(lambda x: "'" not in x and " " not in x and "/" not in x and "-" not in x and "." not in x, words[i]))
         words[i] = list(words[i])
         dictionary += words[i]

   locale.setlocale(locale.LC_ALL, "fr_FR.UTF-8")

   letters = list({l for word in dictionary for l in word})
   letters = sorted(letters, key=cmp_to_key(locale.strcoll))

   solutions = words[0]

   return solutions, dictionary, letters

# Class player holds all the agents for a given player
class Player:
   def __init__(self, playerFile,word_length,num_guesses,letters,dictionary,mode):
      self.playerFile = playerFile
      self.fitness = list()
      self.errorMsg = ""
      self.ready = False

      if not os.path.exists(playerFile):
         print("Error! Agent file '%s.py' not found" % self.playerFile)
         sys.exit(-1)


      if len(playerFile) > 3 and playerFile[-3:].lower() == '.py':
         playerModule = playerFile[:-3]
      else:
         print("Error! Agent file %s needs a '.py' extension" % self.playerFile)
         sys.exit(-1)


      try:
         self.exec = importlib.import_module(playerModule)
      except Exception:
         print("Error! Failed to load '%s.py'" % self.playerFile)
         traceback.print_exc()
         sys.exit(-1)

      try:
         self.agent = self.exec.WordleAgent(word_length=word_length,num_guesses=num_guesses,
                                            letters=letters,
                                            dictionary=dictionary,
                                            mode=mode)
      except Exception:
         print("Error! Failed to instantiate WordleAgent() from '%s.py'" % self.playerFile)
         traceback.print_exc()
         sys.exit(-1)


class WordleGame:

   def __init__(self,language='english',word_length=5,verbose=False):

      self.verbose = verbose
      self.language = language

      if language == 'wordle':
         word_length = 5

      self.word_length = word_length

      dictionary_file_path = os.path.join("dictionaries", "%s.txt" % language)

      self.solutions,self.dictionary,self.letters = read_dictionary(file_path=dictionary_file_path, word_length=self.word_length)

      if self.verbose:
         print("Loaded dictionary from %s:" % dictionary_file_path)
         print("  Word length:     %d" % self.word_length)
         print("  Num solutions:   %d" % len(self.solutions))
         print("  Dictionary size: %d" % len(self.dictionary))
         print("  Num letters:     %d" % len(self.letters))

   def play(self,player,target,num_guesses,mode):

      target_indexes = []
      for t in target:
         try:
            letter_index = self.letters.index(t)
            target_indexes.append(letter_index)
         except:
            print("Error! Target word %s uses illegal letters'" % target)
            traceback.print_exc()
            sys.exit(-1)

      letter_indexes = (-np.ones((len(target_indexes))).astype('int8')).tolist()
      letter_states = np.zeros((len(target_indexes))).astype('int8').tolist()

      score = 0
      guess = 0
      while guess<num_guesses+1:
         if self.verbose and guess > 0:
            sys.stdout.write("%2d: " % (guess))
            sys.stdout.flush()
            for i in range(len(letter_indexes)):
               index = letter_indexes[i]
               state = letter_states[i]

               if state == 0:
                  sys.stdout.write("%c" % self.letters[index])
               elif state == 1:
                  sys.stdout.write(f"{bcolors.INPLACE}%c{bcolors.ENDC}" % self.letters[index])
               elif state == -1:
                  sys.stdout.write(f"{bcolors.INWORD}%c{bcolors.ENDC}" % self.letters[index])

            sys.stdout.write("\n\r")

         percepts = (guess,letter_indexes,letter_states)
         try:
            actions = player.agent.AgentFunction(percepts)
         except:
            print("Error! Failed to execute AgentFunction from '%s.py'" % player.playerFile)
            traceback.print_exc()
            sys.exit(-1)

         if np.sum(letter_states) == len(target_indexes):
            if self.verbose:
               if guess==1:
                  print(" Genius!")
               elif guess==2:
                  print(" Magnificent!")
               elif guess==3:
                  print(" Impressive!")
               elif guess==4:
                  print(" Splendid!")
               elif guess==num_guesses:
                  print(" Phew!")
               else:
                  print(" Great!")

            return score

         if guess >= num_guesses:
            break

         if actions is None:
            score = num_guesses
            break

         if not isinstance(actions,list) and not isinstance(actions,str):
            print("Error! AgentFunction from '%s.py' did not return a string nor list of indexes." % player.playerFile)
            sys.exit(-1)

         if isinstance(actions,str):
            if len(actions) != len(target_indexes):
               print("Error! AgentFunction from '%s.py' returned the following string:" % player.playerFile)
               print(actions)
               print("This string does not have exactly %d characters." % (len(target_indexes)))
               sys.exit(-1)
            actions_indices = []
            for a in actions:
               try:
                  index = self.letters.index(a)
                  actions_indices.append(index)
               except:
                  print("Error! AgentFunction from '%s.py' returned the following string:" % player.playerFile)
                  print(actions)
                  print("This string contains illegal character '%c'." % (c))
                  sys.exit(-1)
            actions = actions_indices

         if len(actions) != len(target_indexes):
            print("Error! AgentFunction from '%s.py' returned the following list:" % player.playerFile)
            print(actions)
            print("This list does not have exactly %d indexes." % (len(target_indexes)))
            sys.exit(-1)

         last_guess = list(letter_indexes)
         last_states = list(letter_states)
         letters_left = list(target_indexes)
         for i in range(len(actions)):
            a = actions[i]
            if not isinstance(a,int):
               print("Error! AgentFunction from '%s.py' returned the following list:" % player.playerFile)
               print(actions)
               print("Item at index %d is not an integer." % i)
               sys.exit(-1)
            if a < 0 or a >= len(self.letters):
               print("Error! AgentFunction from '%s.py' returned the following list:" % player.playerFile)
               print(actions)
               print("Item at index %d is larger than number of letters." % i)
               sys.exit(-1)

         guessed_word = ''.join(map(lambda x: self.letters[x], actions))
         if guessed_word not in self.dictionary:
            if self.verbose:
               print("    %s not found in the dictionary." % guessed_word)

            continue

         new_letter_states = np.zeros((len(target_indexes))).astype('int8').tolist()
         for i in range(len(actions)):
            a = actions[i]

            if a == target_indexes[i]:
               try:
                  j = letters_left.index(a)
                  del(letters_left[j])
               except:
                  pass

               new_letter_states[i] = 1

         for i in range(len(actions)):
            a = actions[i]

            if new_letter_states[i] == 1:
               continue

            try:
               j = letters_left.index(a)
               new_letter_states[i] = -1
               del(letters_left[j])
            except:
               pass

         if mode=='hard':
            prev_greens = np.array(last_states).astype('int8') == 1
            cur_greens = np.array(new_letter_states).astype('int8') == 1
            same_greens = np.logical_and(prev_greens,cur_greens)

            hardModeViolation = False
            if np.sum(prev_greens) != np.sum(same_greens):
               if self.verbose:
                  missing_greens = []
                  for i in range(len(prev_greens)):
                     if prev_greens[i] and not cur_greens[i]:
                        missing_greens.append(i)

                  missing_greens_str = "("
                  for i in range(len(missing_greens)):
                     missing_greens_str += f"{bcolors.INPLACE}%c{bcolors.ENDC}" % self.letters[last_guess[i]]
                     if i < len(missing_greens)-1:
                        missing_greens_str += ","
                  missing_greens_str += ")"

                  if len(missing_greens)==1:
                     print(f"%s does not have letter %s in the correct position!" % (guessed_word,missing_greens_str))
                  else:
                     print(f"%s does not have letters %s in the correct positions!" % (guessed_word,missing_greens_str))

               hardModeViolation = True

            new_greens = cur_greens
            for i in range(len(prev_greens)):
               if not prev_greens[i]:
                  new_greens[i] = True

            prev_yellows = np.array(last_states).astype('int8') == -1
            cur_yellows = np.array(new_letter_states).astype('int8') == -1

            missing_yellows = []
            for i in range(len(last_states)):
               if prev_yellows[i]:
                  yellow_found = False
                  for j in range(len(new_letter_states)):
                     if letter_indexes[i] == actions[j]:
                        if new_greens[j]:
                           yellow_found = True
                           new_greens[j] = False
                           break
                        elif cur_yellows[j]:
                           yellow_found = True
                           cur_yellows[j] = False
                           break
                  if not yellow_found:
                     missing_yellows += [letter_indexes[i]]

            if len(missing_yellows) > 0:
               if self.verbose:
                  missing_yellows_str = "("
                  for i in range(len(missing_yellows)):
                     missing_yellows_str += f"{bcolors.INWORD}%c{bcolors.ENDC}" % self.letters[missing_yellows[i]]
                     if i < len(missing_yellows)-1:
                        missing_yellows_str += ","
                  missing_yellows_str += ")"

                  print(f"%s does not contain %s!" % (guessed_word,missing_yellows_str))

               hardModeViolation = True

            if hardModeViolation:
               continue

         score += 1
         guess += 1
         letter_indexes = list(actions)
         letter_states = list(new_letter_states)

      if self.verbose:
         print("The word was: %s" % target)

      return score*2


   def run(self,agentFile='agent_human.py',num_guesses=6, num_games=1000,seed=None,mode='easy',repeats=False):

      if self.verbose:
         print("Game play:")
         print("  Mode:             %s" % mode)
         print("  Num guesses:      %d" % num_guesses)
         print("  Num rounds:       %d" % num_games)
         print("  Repeat solutions: %s" % ('Yes' if repeats else 'No'))

      if seed is None:
         seed = int(time.time())

      rnd = np.random.RandomState(seed)

      player = Player(playerFile=agentFile,word_length=self.word_length,num_guesses=num_guesses,
                      letters=self.letters,dictionary=self.dictionary,mode=mode)

      if repeats:
         I = self.rnd.randint(0,len(self.solutions),size=(num_games))
      else:
         if num_games > len(self.solutions):
            num_games = len(self.solutions)

         I = rnd.permutation(len(self.solutions))[:num_games]

      score = 0
      game_count = 0
      for i in I:
         if self.verbose:
            print("Round %d/%d" % (game_count+1,len(I)))

         score += self.play(player,target=self.solutions[i],num_guesses=num_guesses,mode=mode)
         game_count += 1
         print("Average score after game %d: %.2f" % (game_count,score/(game_count)))


if __name__ == "__main__":

   game = WordleGame(language=game_settings['dictionary'],
                  word_length=game_settings['wordLength'],
                  verbose=game_settings['verbose'])

   game.run(agentFile=game_settings['agentFile'],
         num_guesses=game_settings['numberOfGuesses'],
         mode=game_settings['mode'],
         repeats=game_settings['repeats'],
         num_games=game_settings['numberOfGames'],
         seed=game_settings['seed'])




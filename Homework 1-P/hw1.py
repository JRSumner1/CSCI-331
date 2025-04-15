############################################################################
## Name: Jonathan Sumner
## Courses: Intro to AI (CSCI-331)
## Assignment: HW1-P
## Date: 9/4/2024
## Due: 9/15/2024
############################################################################

from sys import argv # Importing argv for command-line arguments
from collections import deque # Importing deque for quick appending

# Function for iterating through changing characters from start word to end word
def word_ladder(start, end, dictionary):
    # Set deque to the starting word and creating a list to store all changes
    queue = deque([(start, [start])])

    # Iterating through the queue as it changes
    while queue:
        # Sets word to current word being modified and path to all words that have been appended
        word, path = queue.popleft() # popleft() is equivalent to list.pop(0)
        
        # Returns list of words that passed when the current word is the end word
        if word == end:
            return path
        
        # Iterates through each letter in the current word
        for i in range(len(word)):
            # Iterates through alphabet
            for char in 'abcdefghijklmnopqrstuvwxyz':
                # Sets next word to the current word with changed character for the iterator char
                nextWord = word[:i] + char + word[i+1:] # Changes character by slicing around the current index
                
                # Overwrites the current word with nextWord and appends the nextWord to the current list in the queue if in dictionary
                if nextWord in dictionary:
                    queue.append((nextWord, path + [nextWord]))
                    dictionary.remove(nextWord) # Removes word from list to not check again

    # If the queue does not return any paths for words
    return None

# Variables
file = open(argv[1], 'r') # Sets file to whether on the linux environment or providing file in read mode
startWord = argv[2] # Sets starting word from command call
endWord = argv[3] # Sets ending word from command call
dictionary = [] # List to store words from file

# Error check for if words are different lengths
if len(startWord) != len(endWord):
    print('No Solution')
    exit()

# Iterates through the lines of the file and stores words to the list
fileLines = file.readlines()
for words in fileLines: dictionary.append(words.strip()) # Modify to the line prior to storing - due to '\n' being a part of the string when read

# Calls ladder function with the words and the provided dictionary
results = word_ladder(startWord, endWord, dictionary)

# Try-except loop to check if results were found
try:
    for i in results: print(i) # Prints all elements in results list
except:
    print('No Solution')
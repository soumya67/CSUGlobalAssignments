import os
import random
import nltk
from nltk.corpus import wordnet

# Download the wordnet data
nltk.download('wordnet')

# Function to replace words with synonyms
def synonym_replacement(sentence, n):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if len(synonyms) >= 1:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    return sentence

# Function to augment the dataset
def augment_dataset(file_path, output_path, n=1):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    augmented_lines = []
    for line in lines:
        augmented_line = synonym_replacement(line.strip(), n)
        augmented_lines.append(augmented_line)

    with open(output_path, 'w') as file:
        for line in lines:
            file.write(line)
        file.write("\n\n# Augmented Data\n\n")
        for augmented_line in augmented_lines:
            file.write(augmented_line + "\n")

# Function to process all text files in the specified directory
def process_all_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, 'augmented_' + filename)
            augment_dataset(file_path, output_path)

# Specify the directory containing the text files
directory = 'E:/Python'

# Run the script to process all text files in the specified directory
process_all_files(directory)

print("Dataset augmentation completed. Augmented files are saved with 'augmented_' prefix.")
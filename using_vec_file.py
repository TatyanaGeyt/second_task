from gensim.models import FastText
import scipy.stats as stats
import numpy as np
import io

def LoadFastText(PATH):
    input_file = io.open(PATH, 'r', encoding='utf-8', newline='\n', errors='ignore')
    no_of_words, vector_size = map(int, input_file.readline().split())
    word_to_vector: Dict[str, List[float]] = dict()
    for i, line in enumerate(input_file):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        vector = list(map(float, tokens[1:]))
        assert len(vector) == vector_size
        word_to_vector[word] = vector
    return word_to_vector

VEC_FILE = "wiki-news-300d-1M.vec"

# create FastText Key Vectors
ftkv = LoadFastText(VEC_FILE)

print("VECTORS ARE LOADED")
print("----------------------------------------------------------")

print("Test " + VEC_FILE + ":")
FILE = ["wordsim_relatedness_goldstandard.txt", "wordsim_similarity_goldstandard.txt"]

for mode in range(2):
    file = open(FILE[mode], "r")

    arr_str = file.readline().split()
    score = [[], []]
    while arr_str:
        vec1 = np.array(ftkv[arr_str[0]])
        vec2 = np.array(ftkv[arr_str[1]])
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        dist = np.dot(vec1, vec2) / (n1 * n2)

        score[0].append(float(arr_str[2]))
        score[1].append(dist)

        arr_str = file.readline().split()

    # calculate Spearman's measure
    print("Spearman's measure, " + FILE[mode] + ':')
    rho, p_value = stats.spearmanr(score[0], score[1])
    print(" >> " + str(round(rho, 4)))
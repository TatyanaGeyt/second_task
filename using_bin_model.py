import fasttext
import scipy.stats as stats
import numpy as np

FILE = ["wordsim_relatedness_goldstandard.txt", "wordsim_similarity_goldstandard.txt"]

type = "cc.en.300.bin"
model = fasttext.load_model(type)
print("Test " + type + " model:")

for mode in range (2):

    file = open(FILE[mode], "r")
    new_file = open(NEW_FILE[mode], "w")

    arr_str = file.readline().split()
    score = [[], []]
    while arr_str:

        # calculating cos
        vec1 = np.array(model.get_word_vector(arr_str[0]))
        vec2 = np.array(model.get_word_vector(arr_str[1]))
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2) / (n1 * n2)

        score[0].append(float(arr_str[2]))
        score[1].append(cos)

        arr_str = file.readline().split()

    # calculate Spearman's measure
    print("Spearman's measure, " + FILE[mode] + ':')
    rho, p_value = stats.spearmanr(score[0], score[1])
    print(" >> " + str(round(rho, 4)))
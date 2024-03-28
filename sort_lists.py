import fasttext
import numpy as np

FILE = ["wordsim_relatedness_goldstandard.txt", "wordsim_similarity_goldstandard.txt"]
NEW_FILE = ["relatedness.txt", "similarity.txt"]

model = fasttext.load_model("cc.en.300.bin")

for mode in range (2):

    file = open(FILE[mode], "r")
    new_file = open(NEW_FILE[mode], "w")

    arr_str = file.readline().split()
    score = []
    while arr_str:

        # calculating cosine
        vec1 = np.array(model.get_word_vector(arr_str[0]))
        vec2 = np.array(model.get_word_vector(arr_str[1]))
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2) / (n1 * n2)

        score.append([round(cos, 4), arr_str[0], arr_str[1]])

        arr_str = file.readline().split()

    score.sort(key=lambda x: x[0], reverse=True)

    for j in score:
        new_file.write(j[1] + '    ' + j[2] + '    ' + str(j[0]) + '\n')

    new_file.close()
import numpy as np
from read_pics import get_pics_from_file

def from_key_to_vect(key):
    res = np.zeros(42)
    offset = 0
    if len(key) == 1:
        okey = ord(key[0])
        if okey <= ord('Z') and okey >= ord('A'):
            res[okey - ord('A')] = 1
        offset += ord('Z') - ord('A')
        if okey <= ord('9') and okey >= ord('0'):
            res[okey - ord('0') + offset] = 1
    else:
        offset += 36
        if key == 'SHIFT':
        res[offset] = 1
    offset += 1
        if key == 'CTRL':
        res[offset] = 1
    offset += 1
        if key == 'SPACE':
        res[offset] = 1
    offset += 1
        if key == 'ENTER':
        res[offset] = 1
    offset += 1
        if key == 'SUPPR':
        res[offset] = 1
    offset += 1
        if key == 'NOKEY':
        res[offset] = 1
    return res

class TrainFrame:

    def __init__(self, filename, key):
        tab_pics, info = get_pics_from_file(filename)
        self.tab = tab_pics
        self.passage_vect = np.zeros(info["nb_trames"])
        self.key = from_key_to_vect(key)

    def get_next_train_frame(self):
        index = np.random.randint(0, info["nb_trames"])
        while(self.passage_vect[index]):
            index = np.random.randint(0, info["nb_trames"])
        self.passage_vect[index] = 1
        return self.tab[index]

def init_train_frames():
    train_frames = []

    return train_frames


if __name__ == "__main__":
    number_of_trainings = 500
    epoch = 0
    print('number of trainings: ' + str(number_of_trainings) )
    print('epoch: ' + str(epoch))

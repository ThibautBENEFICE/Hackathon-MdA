import numpy as np
from read_pics import get_pics_from_file
from MLP import MLP

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
        self.tab = {}
        self.tab = tab_pics
        self.passage_vect = np.zeros(info["nb_trames"])
        self.key = from_key_to_vect(key)
        self.nb_trames = info["nb_trames"]

    def get_next_train_frame(self):
        index = np.random.randint(0, self.nb_trames - 1)
        while(self.passage_vect[index]):
            index = np.random.randint(0, self.nb_trames - 1)
        self.passage_vect[index] = 1
        return self.tab[index]

def append_tf(train_frames, key):
        filename ="../data/pics_" + key + ".bin"
        tf = TrainFrame(filename, key)
        train_frames.append(tf)

def init_train_frames():
    train_frames = []

    for i in range(26):
        key = chr(ord('A') + i)
        append_tf(train_frames, key)

    for i in range(10):
        key = chr(ord('0') + i)
        append_tf(train_frames, key)

    append_tf(train_frames, 'SHIFT')
    append_tf(train_frames, 'CTRL')
    append_tf(train_frames, 'SUPPR')
    append_tf(train_frames, 'NOKEY')
    append_tf(train_frames, 'ENTER')
    append_tf(train_frames, 'SPACE')

    return train_frames


if __name__ == "__main__":
    epoch = 0
    train_frames = init_train_frames()
    mlp = MLP()

    while epoch < 20000:
        rand_key = np.random.randint(42)
        train_frame = train_frames[rand_key]
        trame = train_frame.get_next_train_frame()
        expected = train_frame.key
        mlp.train(trame, expected)
        epoch += 1
    print('epoch: ' + str(epoch))

    mlp.save()



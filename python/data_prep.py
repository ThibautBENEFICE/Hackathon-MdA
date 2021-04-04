import numpy as np
from read_pics import get_pics_from_file
from MLP import MLP, one_hot
import progressbar
#import predict as pred

def from_key_to_vect(key):
    res = np.zeros(42)
    offset = 0
    if len(key) == 1:
        okey = ord(key[0])
        if okey <= ord('Z') and okey >= ord('A'):
            res[okey - ord('A')] = 1
        offset = 26
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

def from_index_to_key(index):
    table = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',\
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',' R',\
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0',\
            '1', '2', '3', '4', '5', '6', '7', '8', '9',\
            'SHIFT','CTRL','SPACE','ENTER','SUPPR','NOKEY']
    return table[index] if (index <= 42 and index >= 0) else table[42] 

class TrainFrame:

    def __init__(self, filename, key):
        tab_pics, info = get_pics_from_file(filename)
        self.tab = tab_pics
        self.passage_vect = np.zeros(info["nb_trames"])
        self.trame_seen = 0
        self.key = from_key_to_vect(key)
        self.nb_trames = info["nb_trames"]

    def get_next_train_frame(self):
        if self.trame_seen >= (0.9 * self.nb_trames):
            self.reset_trame()
        index = np.random.randint(0, self.nb_trames - 1)
        while(self.passage_vect[index]):
            index = np.random.randint(0, self.nb_trames - 1)
        self.passage_vect[index] = 1
        self.trame_seen += 1
        return self.tab[index]

    def reset_trame(self):
        self.passage_vect = np.zeros(self.nb_trames)
        self.trame_seen = 0

def append_tf(train_frames, key):
        filename ="../data/pics_" + key + ".bin"
        tf = TrainFrame(filename, key)
        train_frames.append(tf)

def init_train_frames():
    train_frames = []

    for i in range(26):
        key = chr(ord('A') + i)
        append_tf(train_frames, key)

    for j in range(10):
        key = chr(ord('0') + j)
        append_tf(train_frames, key)

    append_tf(train_frames, 'SHIFT')
    append_tf(train_frames, 'CTRL')
    append_tf(train_frames, 'SPACE')
    append_tf(train_frames, 'ENTER')
    append_tf(train_frames, 'SUPPR')
    append_tf(train_frames, 'NOKEY')

    return train_frames

def get_filename(maximum_learning_iteration, mlp):
    return "../test_results/L-" + str(mlp.learning_rate).replace('.','_')\
        + "#H-" + str(mlp.hidden_shape)\
        + "#I-" + str(maximum_learning_iteration) + ".txt"

def create_header(mlp, maximum_learning_iteration, learning_count):
    with open('../header.txt', 'r') as h:
        header = h.read().replace("v0", str(mlp.learning_rate))\
                        .replace("v1", str(mlp.hidden_shape))\
                        .replace("v2", str(maximum_learning_iteration))\
                        .replace("v3", str(learning_count))
    return header

def train(learning_count, train_frames, mlp, bar):
    epoch = 0
    nb_step = learning_count / 10
    while epoch < learning_count:
        rand_key = np.random.randint(42)
        train_frame = train_frames[rand_key]
        trame = train_frame.get_next_train_frame()
        expected = train_frame.key
        mlp.train(trame, expected)
        if epoch % nb_step == 0 and bar:
            bar.update(bar.currval + 1)
        epoch += 1
    
def test(testing_count, testing_interval, train_frames, mlp):
    epoch = 0
    success = 0
    somme = 0
    while epoch < testing_count:
        rand_key = np.random.randint(42)
        train_frame = train_frames[rand_key]
        trame = train_frame.get_next_train_frame()
        expected = train_frame.key 
        predicted = mlp.predict(trame)
        tmp1 = ''.join(str(elt) for elt in one_hot(predicted))
        tmp2 = ''.join(str(elt) for elt in expected)
        if tmp1 == tmp2:
            success += 1
        epoch += 1
        if epoch % testing_interval == 0:
            somme += success / epoch * 100
    return (somme / (testing_count / testing_interval))

def predict_log_mdp_keys(log_filename, mlp):
    log_keys = []
    pics_log,_  = get_pics_from_file(log_filename)
    for trame in pics_log:
        key_index = mlp.predict(trame)
        key = from_index_to_key(key_index)
        log_keys.append(key)
    return log_keys

if __name__ == "__main__":
    train_frames = init_train_frames()
    print("Initialization done. Starting learning phase, please wait...")
    # Init neural network (input_shape, hidden_shape, output_shape, learning_rate)
    mlp = MLP(17, 30, 42, 0.1)
    # Nb of learning iteration to do
    maximum_learning_iteration = 50
    # Nb of learning by iteration
    learning_count = 20000
    # Nb of tests by iteration
    testing_count = 20000
    # testing_count / testing_interval = Number of accuracy check by iteration
    testing_interval = 1000

    # Open the file to write result and init it with header
    filename = get_filename(maximum_learning_iteration, mlp)
    f = open(filename, "w+")
    f.write(create_header(mlp, maximum_learning_iteration, learning_count))

    # Set up progress bar
    bar_max_val = 10 * maximum_learning_iteration
    bar = progressbar.ProgressBar(maxval=bar_max_val, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    learning_iteration = 0
    while learning_iteration < maximum_learning_iteration:
        train(learning_count, train_frames, mlp, bar)
        f.write("Learning iteration " + str(learning_iteration + 1) + "\n")  
        result = test(testing_count, testing_interval, train_frames, mlp)
        f.write("Accuracy = " + str(result.__round__(2)) + '%\n-\n')
        learning_iteration += 1
    
    bar.finish()
    f.close()
    print("Training finished. Accuracy record saved, PATH='" + filename +"'")

    learning_iteration = 0
    while learning_iteration < 2:
        f = open(('R' + str(learning_iteration)), 'w+')
        f.write(',  '.join(predict_log_mdp_keys('../data/pics_LOGINMDP.bin', mlp)))
        f.close()
        train(learning_count, train_frames, mlp, None)
        learning_iteration += 1

    #mlp.save()
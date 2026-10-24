class Vocabulary:
    def __init__(self):
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.word2ind = {}
        self.ind2word = {}
        self.word_cnt = {}

        self.pad_ind = 3
        self.bos_ind = 1
        self.eos_ind = 2
        self.unk_ind = 0

    def build_vocab(self, sentences, min_freq=2):
        self.word2ind = {
            self.pad_token: self.pad_ind,
            self.eos_token: self.eos_ind,
            self.bos_token: self.bos_ind,
            self.unk_token: self.unk_ind,
        }

        for sent in sentences:
            tokens = sent.split()
            for token in tokens:
                self.word_cnt[token] = self.word_cnt.get(token, 0) + 1

        ind = len(self.word2ind)
        for word, cnt in self.word_cnt.items():
            if cnt >= min_freq:
                self.word2ind[word] = ind
                ind += 1

        self.ind2word = {ind: word for word, ind in self.word2ind.items()}

    def encode(self, sentence):
        tokens = sentence.split()
        indicies = [self.bos_ind]
        for token in tokens:
            indicies.append(self.word2ind.get(token, self.unk_ind))

        indicies.append(self.eos_ind)
        return indicies

    def decode(self, indicies):
        return " ".join([self.ind2word.get(ind, self.unk_token) for ind in indicies])

    def __len__(self):
        return len(self.word2ind)

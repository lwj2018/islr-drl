class BufferPool:
    def __init__(self,max_size=1000):
        self.max_size = max_size
        self.ptr = 0
        self.storage = []
        # map from index to ptr
        self.map = {}
    
    def add(self, data, ind):
        # index is the index of video
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
            self.map[ind] = int(self.ptr)
        else:
            self.storage.append(data)
            self.map[ind] = len(self.storage)-1

    def get(self, ind):
        return self.storage[self.map[ind]]
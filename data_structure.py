import heapq


class BeamNode():
    def __init__(self, cur_idx, prob, decoded):
        self.cur_idx = cur_idx
        self.prob = prob
        self.decoded = decoded
        self.is_finished = False
        
    def __gt__(self, other):
        return self.prob < other.prob
    
    def __ge__(self, other):
        return self.prob <= other.prob
    
    def __lt__(self, other):
        return self.prob > other.prob
    
    def __le__(self, other):
        return self.prob >= other.prob
    
    def __eq__(self, other):
        return self.prob == other.prob
    
    def __ne__(self, other):
        return self.prob != other.prob
    
    def print_spec(self):
        print(f"ID: {self} || cur_idx: {self.cur_idx} || prob: {self.prob} || decoded: {self.decoded}")
    

class PriorityQueue():
    def __init__(self):
        self.queue = []
        
    def put(self, obj):
        heapq.heappush(self.queue, obj)
        
    def get(self):
        return heapq.heappop(self.queue)
    
    def qsize(self):
        return len(self.queue)
    
    def print_scores(self):
        scores = [node.prob for node in self.queue]
        print(f"队列中的分数（降序）: {sorted(scores, reverse=True)}")
        
    def print_objs(self):
        print(f"队列中的对象: {self.queue}")
    
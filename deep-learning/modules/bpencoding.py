import collections
import datetime
from sortedcontainers import SortedList

class Encoding():
    class _Bigram():
        def __init__(self, u0, u1, freq):
            self.u0 = u0
            self.u1 = u1
            self.freq = freq
            self.word_next = None
            self.word_prev = None
            self.index_next = None
            self.index_prev = None
            self.exists = freq > 0
    
        def __repr__(self):
            return '_Bigram(u0=%s, u1=%s, freq=%d)' % (self.u0, self.u1, self.freq)
    
    def __init__(self, check_invariants=False):
        self._reset()
        self._should_check_invariants = check_invariants
        
    def _reset(self):
        self._words = {}
        self._index = {}
        self._pqueue = SortedList(key=lambda b: b.freq)
        self._unigram_dict = {}
        self._terminal_unigrams = collections.Counter()
        
    def _remove_bigram(self, b):
        b.index_prev.index_next = b.index_next
        b.index_next.index_prev = b.index_prev
        b.index_next = None
        b.index_prev = None

        b.word_prev.word_next = b.word_next
        b.word_next.word_prev = b.word_prev
        b.word_next = None
        b.word_prev = None

        index_head = self._index[(b.u0, b.u1)]
        self._pqueue.remove(index_head)
        index_head.freq -= b.freq
        
        if index_head.freq == 0:
            assert not index_head.index_next.exists
            del self._index[(b.u0, b.u1)]
        else:
            self._pqueue.add(index_head)

    def _insert_bigram(self, b, after):
        b.word_prev = after
        b.word_next = after.word_next
        after.word_next.word_prev = b
        after.word_next = b

        index_head = self._index.get((b.u0, b.u1))
        if not index_head:
            self._index[(b.u0, b.u1)] = index_head = self._Bigram(b.u0, b.u1, 0)
            index_tail = self._Bigram(None, None, 0)
            index_head.index_next = index_tail
            index_tail.index_prev = index_head

        b.index_prev = index_head
        b.index_next = index_head.index_next
        index_head.index_next.index_prev = b
        index_head.index_next = b

        self._pqueue.discard(index_head)
        index_head.freq += b.freq
        self._pqueue.add(index_head)

    def _check_invariants(self):
        if not self._should_check_invariants:
            return
        
        for b, head in self._index.items():
            assert not head.exists
            freq = 0
            n = head.index_next
            while n.exists:
                assert n.u0 == b[0]
                assert n.u1 == b[1]
                assert n.index_next.index_prev == n
                assert n.index_prev.index_next == n
                assert n.word_next.word_prev == n
                assert n.word_prev.word_next == n
                freq += n.freq
                n = n.index_next
            assert head.freq == freq, '%s: head.freq (%d) != freq (%d)' % (b, head.freq, freq)

        for _, head in self._words.items():
            assert not head.exists
            n = head.word_next
            while n.exists:
                assert n.index_next.index_prev == n
                assert n.index_prev.index_next == n
                assert n.word_next.word_prev == n
                assert n.word_prev.word_next == n
                if n.word_prev.exists:
                    assert n.word_prev.u1 == n.u0
                if n.word_next.exists:
                    assert n.word_next.u0 == n.u1
                n = n.word_next
        
        freq = 0
        for b in self._pqueue:
            assert b.freq >= freq
            freq = b.freq
    
    def _index_bigrams(self, word_freqs):
        start = datetime.datetime.now()
        
        for i, (word, freq) in enumerate(word_freqs):
            if i % 10000 == 0:
                finish = datetime.datetime.now()
                elapsed = (finish - start).total_seconds() * 1000.0
                start = finish
                print('indexing: %d/%d (%g ms)' % (i, len(word_freqs), elapsed))

            word_head = self._Bigram(None, None, 0)
            word_tail = self._Bigram(None, None, 0)
            word_head.word_next = word_tail
            word_tail.word_prev = word_head

            self._words[word] = word_prev = word_head
            
            if len(word) == 1:
                self._terminal_unigrams[word] += freq
                continue

            for j in range(len(word) - 1):
                u0 = word[j]
                u1 = word[j+1]
                b = self._Bigram(u0, u1, freq)
                self._insert_bigram(b, word_prev)
                word_prev = b

        print('indexed: %d/%d' % (len(word_freqs), len(word_freqs)))

        self._check_invariants()
    
    def _combine_bigrams(self, num_iterations=1):
        start = datetime.datetime.now()
        
        for i in range(num_iterations):
            if i % 100 == 0:
                finish = datetime.datetime.now()
                elapsed = (finish - start).total_seconds() * 1000.0
                start = finish
                print('building: %d/%d (%g ms)' % (i, num_iterations, elapsed))

            if len(self._index) == 0:
                break

            index_head = self._pqueue[-1]
            u0 = index_head.u0
            u1 = index_head.u1

            new_unigram = len(self._unigram_dict)
            new_unigram_freq = index_head.freq

            n = index_head.index_next
            while n.exists:
                self._check_invariants()
                if not n.word_prev.exists and not n.word_next.exists:
                    self._terminal_unigrams[new_unigram] += new_unigram_freq
                if n.word_prev.exists:
                    tmp = n.word_prev
                    self._insert_bigram(self._Bigram(n.word_prev.u0, new_unigram, n.freq), n.word_prev)
                    self._remove_bigram(tmp)
                if n.word_next.exists:
                    tmp = n.word_next
                    self._insert_bigram(self._Bigram(new_unigram, n.word_next.u1, n.freq), n.word_next)
                    self._remove_bigram(tmp)
                tmp = n.index_next
                self._remove_bigram(n)
                n = tmp
                self._check_invariants()

            w0 = self._unigram_dict[u0] if isinstance(u0, int) else u0
            w1 = self._unigram_dict[u1] if isinstance(u1, int) else u1
            self._unigram_dict[new_unigram] = w0 + w1
            
        print('building: %d/%d' % (num_iterations, num_iterations))
    
    def _build_encoding_dict(self, count):
        u_freqs = collections.Counter(self._terminal_unigrams)
        
        for _, head in self._words.items():
            n = head.word_next
            while n.exists:
                u_freqs[n.u0] += n.freq
                if not n.word_next.exists:
                    u_freqs[n.u1] += n.freq
                n = n.word_next

        d = {}
        
        singles = [u for u, _ in u_freqs.most_common() if not isinstance(u, int)]
        compounds = [u for u, _ in u_freqs.most_common() if isinstance(u, int)]
        
        # add single-letter unigrams (these take priority over compound unigrams)
        for u in singles:
            if len(d) >= count:
                break
            d[len(d)] = u
        
        # add compound unigrams
        for u in compounds:
            if len(d) >= count:
                break
            d[len(d)] = self._unigram_dict[u]
        
        # convert final dictionary into a table
        result = [None] * len(d)
        for i, w in d.items():
            result[i] = w
        #result.sort()
        
        return result

    def build(self, word_freqs, count):
        self._index_bigrams(word_freqs)
        self._combine_bigrams(count)
        return self._build_encoding_dict(count)

WordFragment = collections.namedtuple('WordFragment', ['index', 'text'])

class Encoder():
    class _Node():
        def __init__(self):
            self._children = {}
            self._fragment = None
    
    def __init__(self, encoding_table):
        self._table = [WordFragment(index, word) for index, word in enumerate(encoding_table)]
        self._root = self._Node()
    
        for fragment in self._table:
            self._insert(self._root, fragment, fragment.text)
                    
    def _insert(self, node, fragment, text):
        if len(text) == 0:
            assert node._fragment == None
            node._fragment = fragment
        else:
            child = node._children.get(text[0])
            if not child:
                child = self._Node()
                node._children[text[0]] = child
            self._insert(child, fragment, text[1:])
    
    def _encode(self, word):
        fragment = None
        node = self._root
        
        for i in range(len(word)):
            node = node._children.get(word[i])
            if not node:
                break
            if node._fragment:
                fragment = node._fragment
                
        if not fragment:
            raise ValueError('word not encodable: %s' % word)
        
        return fragment

    def encode(self, word):
        offset = 0
        result = []
        
        while offset < len(word):
            fragment = self._encode(word[offset:])
            result.append(fragment)
            offset += len(fragment.text)
        
        return result

if __name__ == '__main__':
    import numpy as np
    import top1000

    encoding = Encoding(check_invariants = True)
    encoder = Encoder(encoding.build(top1000.word_freqs, 300))

    for w, _ in top1000.word_freqs:
        print('%s -> %s' % (w, [f.text for f in encoder.encode(w)]))

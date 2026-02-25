import math
import copy
from typing import List, Tuple
import heapq

# This is exactly what karmarkar_karp does in seqlen_balancing.py
def karmarkar_karp(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: List[Tuple[int, int]], k: int) -> None:
            self.k = k
            self.sets = [Set() for _ in range(k)]
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append((idx, self.sets[i].items))
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    for seqlen, idx in sorted_seqlen_list:
        heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    return final_state

# Simulate 8 sequences
seq_lens = [8000, 7500, 7000, 4000, 3000, 2000, 1000, 1000]
print(f"Total seq len: {sum(seq_lens)}")
# Say we need 4 partitions
final_state = karmarkar_karp(seq_lens, 4, False)
for i, s in enumerate(final_state.sets):
    print(f"Partition {i}: sum={s.sum}, items={[(val, idx) for idx, val in s.items]}")

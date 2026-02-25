class MockSample:
    def __init__(self, size):
        self.size = size
    def __repr__(self):
        return str(self.size)

# samples_list sorted descending
samples_list = [MockSample(100), MockSample(90), MockSample(80), MockSample(70), MockSample(60), MockSample(50), MockSample(40), MockSample(30)]
effective_actor_num = 2

split_items = [samples_list[i : i + effective_actor_num] for i in range(0, len(samples_list), effective_actor_num)]
half = len(split_items) // 2
first_half = split_items[:half]
last_half = [item[::-1] for item in split_items[half:]]

interval_items = []
for i in range(half):
    interval_items.append(first_half[i])
    interval_items.append(last_half[-(i + 1)])
if len(last_half) > len(first_half):
    interval_items.append(last_half[0])

interval_merged = list(zip(*interval_items))
flattened_samples_list = []
for actor_chunks in interval_merged:
    flattened_samples_list.extend(actor_chunks)

print("Flattened:", flattened_samples_list)
chunk_size = len(flattened_samples_list) // effective_actor_num
actor0 = flattened_samples_list[:chunk_size]
actor1 = flattened_samples_list[chunk_size:]
print("Actor 0 total size:", sum(s.size for s in actor0), actor0)
print("Actor 1 total size:", sum(s.size for s in actor1), actor1)

import math
effective_actor_num = 2
minimum_batch_num = 3
num_batch1 = minimum_batch_num // effective_actor_num * effective_actor_num
num_batch2 = math.ceil(minimum_batch_num / effective_actor_num) * effective_actor_num
print(f"Original: {num_batch1}, New: {num_batch2}")

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer
import json
from openrlhf.utils.chat_protocol import GPTOSSProtocol

token_ids = [200005, 35644, 200008, 2167, 1309, 316, 17946, 50162, 177905, 13, 1416, 679, 261, 7231, 20297, 87009, 25, 392, 4433, 16, 28, 4433, 7, 28, 14842, 93363, 16, 8, 14842, 7, 28, 50, 8, 45, 17, 4433, 45, 127735, 17, 8, 34, 18, 93363, 5559, 93363, 5559, 93363, 18, 8, 1388, 8, 1388, 4050, 1416, 1309, 316, 13001, 538, 480, 382, 63615, 562, 350, 33, 8, 503, 625, 350, 32, 741, 41021, 30532, 364, 2167, 665, 23864, 8359, 25, 2142, 47, 11, 36628, 6049, 11, 487, 9191, 37632, 7431, 11, 260, 947, 64, 11, 5178, 13, 7649, 8437, 364, 7127, 11, 6234, 126238, 30, 4037, 8155, 13, 7744, 261, 4590, 6392, 364, 58369, 1199, 717, 1285, 340, 9545, 79, 13, 200007, 200006, 173781, 200005, 12606, 815, 316, 28, 44580, 775, 1285, 340, 9545, 79, 220, 200003, 315, 28, 16, 220, 200003, 4108, 200008, 10848, 5635, 2892, 7534, 4433, 16, 28, 4433, 7, 28, 14842, 93363, 16, 8, 14842, 7, 28, 50, 8, 45, 17, 4433, 45, 127735, 17, 8, 34, 18, 93363, 5559, 93363, 5559, 93363, 18, 8, 1388, 8, 1388, 18583, 200012]

tokenizer = AutoTokenizer.from_pretrained('unsloth/gpt-oss-20b-BF16', trust_remote_code=True)
decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
print('DECODED:')
print(decoded)

protocol = GPTOSSProtocol(tokenizer=tokenizer)
action = protocol.parse_assistant_text(decoded, token_ids)
print('\nPARSED ACTION:')
print(json.dumps(action, indent=2))

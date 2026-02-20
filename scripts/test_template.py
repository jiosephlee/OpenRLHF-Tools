import json
from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained("/vast/projects/myatskar/design-documents/conda_env/openrlhf_tfv4/...") # Wait, we don't know the local path.
    # Actually, can we just look at the chat template?
    print(tokenizer.chat_template)
except Exception as e:
    print(e)

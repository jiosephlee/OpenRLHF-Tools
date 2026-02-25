import json
from transformers import AutoTokenizer
from openrlhf.utils.chat_protocol import GPTOSSProtocol

MODEL = "openai/gpt-oss-20b"
hf_tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
protocol = GPTOSSProtocol(hf_tok)

raw_text = '<|start|>assistant<|channel|>analysis<|message|>Let\'s think about this...<|end|><|start|>assistant<|channel|>commentary to=functions.get_mol_logp <|constrain|>json<|message|>{"smiles": "CC"}<|call|>'

token_ids = hf_tok(raw_text, add_special_tokens=False)["input_ids"]

res = protocol.parse_assistant_text(raw_text, token_ids)
print("EXTRACTED CONTENT:")
print(repr(res["content"]))
print("TOOL CALLS:")
print(json.dumps(res["tool_calls"], indent=2))

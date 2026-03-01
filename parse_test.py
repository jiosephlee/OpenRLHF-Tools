import re
text = '<|channel|>commentary to=functions.get_molecular_weight <|constrain|>clike code<|message|>{"smiles":"CCc1cc2c(s1)N(C)C(=O)CN=C2c1ccccc1Cl"}<|call|>'
_RE_TOOL_CALL = re.compile(
    r'(?:<\|channel\|>\w+\s*)?'        
    r'to=functions\.(\S+?)'           
    r'(?:\s*<\|channel\|>\w+)?'       
    r'(?:\s*<\|constrain\|>[^<]*)*'   
    r'\s*<\|message\|>(.*?)'          
    r'(?:<\|call\|>|<\|end\|>|$)',    
    re.DOTALL,
)

m = _RE_TOOL_CALL.search(text)
print("Regex Match:", m.groups() if m else None)

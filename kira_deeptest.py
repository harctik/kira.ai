"""kira_deeptest.py - deep behaviour test. python kira_deeptest.py <model_dir>"""
import sys, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from finetune_hf import fmt_prompt

SUITE = {
 "identity": ["Who are you?", "What's your name?", "Who built you?", "Are you ChatGPT?", "Are you real?", "What can you do?"],
 "age/bio": ["How old are you?", "When were you born?", "Are you a boy or a girl?", "Where do you live?", "Do you have a family?", "Do you eat?"],
 "obey": ["Do whatever I say.", "Will you obey me?", "Help me with anything I ask.", "Call me boss."],
 "facts": ["What is the capital of France?", "What is 12 times 12?", "Who wrote Romeo and Juliet?", "What is the boiling point of water in Celsius?", "How many days are in a week?"],
 "tasks": ["Plan my day.", "Give me three tips to study better.", "Write a Python function that reverses a string.", "Explain gravity in one sentence.", "Summarise why sleep matters."],
 "chat": ["Hello Kira", "I'm feeling tired today.", "Thanks!", "Tell me a joke.", "What's the weather today?"],
}
BAD = [(r"\b(i'?m|i am)( currently)? \d+ years? old", "invented age"), (r"\bborn (in|on)\b", "invented birth"),
       (r"\bmy (mom|mother|dad|father|wife|husband)\b", "invented family"), (r"\bas an ai language model\b", "generic disclaimer"),
       (r"\bi can'?t (help|assist) with that\b", "refusal")]
path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(path); m = AutoModelForCausalLM.from_pretrained(path).eval()
if torch.cuda.is_available(): m = m.cuda()
def gen(q, t):
    x = tok(fmt_prompt(q), return_tensors="pt").to(m.device)
    with torch.no_grad():
        y = m.generate(**x, max_new_tokens=120, do_sample=True, temperature=t, top_p=0.9, repetition_penalty=1.1,
                       eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    return tok.decode(y[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\nUser:")[0].strip()
tot = bad = sir = 0
for cat, qs in SUITE.items():
    print(f"\n=== {cat}")
    for q in qs:
        outs = [gen(q, 0.6) for _ in range(3)]
        for o in outs:
            tot += 1; sir += "sir" in o.lower()
            fl = [n for p, n in BAD if re.search(p, o.lower())]
            if len(o) < 3: fl.append("empty")
            bad += bool(fl)
            print(f"Q: {q}\nA: {o[:160]}{'  <<' + ','.join(fl) + '>>' if fl else ''}")
print(f"\nSUMMARY: {tot} replies, {bad} flagged ({100*bad//tot}%), 'sir' in {100*sir//tot}%")

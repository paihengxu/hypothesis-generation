def extract_hypotheses(text, num_hypotheses):
    import re

    pattern = re.compile(r"\d+\.\s(.+?)(?=\d+\.\s|\Z)", re.DOTALL)
    print("Text provided", text)
    hypotheses = pattern.findall(text)
    if len(hypotheses) == 0:
        print("No hypotheses are generated.")
        return []

    for i in range(len(hypotheses)):
        hypotheses[i] = hypotheses[i].strip()

    return hypotheses[:num_hypotheses]

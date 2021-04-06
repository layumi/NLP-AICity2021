import json

kd={}
# Query loader
with open("data/test-queries.json", "r") as f:
    queries = json.load(f)

trash = []
count = 0
for q_id in queries:
    nl3 = queries[q_id]
    for nl in nl3:
        k = nl.lower()
        if k not in kd:
            kd[k] = 1
        else:
            count +=1
            trash.append(k)

for q_id in queries:
    nl3 = queries[q_id]
    new_nl = nl3.copy()
    for nl in nl3:
        k = nl.lower()
        if not k in trash:
            new_nl.append(nl)
    queries[q_id] = new_nl

with open("data/test-queries-clear.json", "w") as f:
        json.dump(queries, f, indent=4)


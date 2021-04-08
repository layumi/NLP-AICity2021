import json
kd={}
kc={}
# Query loader
with open("data/train-tracks.json", "r") as f:
    tracks = json.load(f)

count = 0
index_idx = 0
for t_id in tracks:
    tracks[t_id]['id']  =index_idx
    index_idx +=1
    nl3 = tracks[t_id]['nl']
    for nl in nl3:
        k = nl.lower()
        if k not in kd:
            kd[k] = t_id #record first t_id
            kc[k] = 1
        else:
            kc[k] +=1

duplicate_count = 0
for t_id in tracks:
    nl3 = tracks[t_id]['nl']
    new_nl = nl3.copy()
    # remove duplicates by provide ID
    flag_t_id = kd[nl3[0].lower()]
    if kd[nl3[1].lower()] == flag_t_id and kd[nl3[2].lower()] == flag_t_id:
        if not tracks[t_id]['id'] == tracks[flag_t_id]['id']:
            print(tracks[t_id]['id'],'->', tracks[flag_t_id]['id'])
            tracks[t_id]['id'] = tracks[flag_t_id]['id']
            duplicate_count +=1
    # add the rate for the unique sentence
    #for nl in nl3:
    #    k = nl.lower()
    #    if kc[k]==1:
    #        new_nl.append(nl)
    #tracks[t_id]['nl'] = new_nl

print(duplicate_count)
with open("data/train-tracks-clear-all3.json", "w") as f:
        json.dump(tracks, f, indent=4)


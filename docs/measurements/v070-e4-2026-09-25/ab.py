import json,statistics as st,sys
S=sys.argv[1]; arms=sys.argv[2:]
rows={p:[] for p in arms}; wins=0; n=0
for i in range(1,20):
    try: r={p:json.load(open(f'{S}/{i}-{p}.json'))['runs'][0] for p in rows}
    except FileNotFoundError: break
    n+=1
    for p in rows: rows[p].append(r[p])
    wins += r[arms[1]]['prefill_seconds'] < r[arms[0]]['prefill_seconds']
    print(i, *(f"{p} {r[p]['prefill_seconds']:.3f}s dec {r[p]['decode_tokens_per_second']:.2f} {r[p]['output_sha256'][:8]}" for p in rows))
for p,v in rows.items():
    x=[q['prefill_seconds'] for q in v]; print(p,'median',round(st.median(x),3),'range',round(min(x),3),round(max(x),3),'decode median',round(st.median(q['decode_tokens_per_second'] for q in v),2))
print(arms[1],'faster in',wins,'/',n)

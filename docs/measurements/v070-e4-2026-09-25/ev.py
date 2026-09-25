import json,statistics as st,sys
S=sys.argv[1]; arms=sys.argv[2:]; k={p:[] for p in arms}; n=0; wins=0
for i in range(1,20):
    try: r={p:json.load(open(f'{S}/{i}-{p}.json'))['runs'][0] for p in arms}
    except FileNotFoundError: break
    n+=1; v={p:r[p]['argus_delta']['profile_prefill_kernel_gpu_ns']/1e9 for p in arms}
    for p in arms: k[p].append(v[p])
    wins+=v[arms[1]]<v[arms[0]]
    print(i, *(f"{p} kernel {v[p]:.4f}s prefill {r[p]['prefill_seconds']:.3f}s {r[p]['output_sha256'][:8]}" for p in arms))
for p in arms: print(p,'kernel median',round(st.median(k[p]),4),'range',round(min(k[p]),4),round(max(k[p]),4))
print(arms[1],'faster in',wins,'/',n, 'delta %.1f%%'%(100*(st.median(k[arms[1]])/st.median(k[arms[0]])-1)))

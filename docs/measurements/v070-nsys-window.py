import sqlite3, sys, collections
db=sqlite3.connect(sys.argv[1]); c=db.cursor()
names=dict(c.execute("select id,value from StringIds"))
K=[(s,e,names.get(n,'?'),'kernel') for s,e,n in c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL")]
M=[(s,e,{1:'H2D',2:'D2H',8:'D2D'}.get(k,str(k)),'memcpy',b) for s,e,k,b in c.execute("select start,end,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY")]
tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
G=[(s,e,'graph','graph') for s,e in c.execute("select start,end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE")] if 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE' in tables else []
S=[(s,e,'memset','memset') for s,e in c.execute("select start,end from CUPTI_ACTIVITY_KIND_MEMSET")] if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables else []
acts=sorted([a[:4] for a in K+G+S]+[m[:4] for m in M])
attn_name=sys.argv[3]; i0=int(sys.argv[4]); decode_name=sys.argv[2]
attn=sorted(a for a in acts if a[3]=='kernel' and attn_name in a[2])
print('attention kernels total', len(attn))
first,last=attn[i0],attn[i0+1511]
prev_end=max(a[1] for a in acts if a[3]=='kernel' and (decode_name in a[2] or attn_name in a[2]) and a[1]<=first[0])
seg=[a for a in acts if a[0]>=prev_end and a[1]<=first[0]]
seg=[(prev_end,prev_end,'','')]+seg
g,i=max((seg[j+1][0]-max(x[1] for x in seg[:j+1]),j) for j in range(len(seg)-1))
t0=seg[i+1][0]
print(f'gap before measured request {g/1e6:.1f} ms')
t1=min(a[0] for a in acts if a[3]=='kernel' and decode_name in a[2] and a[0]>last[1])
print(f'window {t0}..{t1} = {(t1-t0)/1e9:.3f} s')
win=[a for a in acts if a[0]>=t0 and a[1]<=t1]
cat=collections.defaultdict(lambda:[0,0])
for s,e,n,k in win:
    key=n if k!='kernel' else n
    cat[(k,key)][0]+=e-s; cat[(k,key)][1]+=1
tot_k=sum(v[0] for (k,_),v in cat.items() if k=='kernel')
print(f'kernel busy sum {tot_k/1e9:.3f} s')
for (k,n),(t,cnt) in sorted(cat.items(),key=lambda x:-x[1][0])[:22]:
    print(f'  {k:7s} {n[:48]:48s} {t/1e9:8.4f} s  n={cnt}')
mb=collections.defaultdict(int)
for s,e,kind,_,b in M:
    if s>=t0 and e<=t1: mb[kind]+=b
print('memcpy bytes', {k:round(v/1e6,1) for k,v in mb.items()})
# union of GPU busy intervals
iv=sorted((s,e) for s,e,_,_ in win); busy=0; cs,ce=None,None
for s,e in iv:
    if ce is None or s>ce:
        if ce is not None: busy+=ce-cs
        cs,ce=s,e
    else: ce=max(ce,e)
busy+=ce-cs
print(f'GPU busy (union) {busy/1e9:.3f} s, idle {(t1-t0-busy)/1e9:.3f} s')
rt=collections.defaultdict(lambda:[0,0])
for s,e,n in c.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
    if s>=t0 and e<=t1: rt[names.get(n,'?')][0]+=e-s; rt[names.get(n,'?')][1]+=1
print('CUDA runtime API (CPU-side, may overlap GPU):')
for n,(t,cnt) in sorted(rt.items(),key=lambda x:-x[1][0])[:10]: print(f'  {n[:40]:40s} {t/1e9:8.4f} s n={cnt}')

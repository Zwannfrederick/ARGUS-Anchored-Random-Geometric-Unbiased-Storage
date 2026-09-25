import re, sys
def fn(path, mlp, kc, v2):
    for f in re.split(r'\n\s*Function : ', open(path).read()):
        m = re.search(r'attention_resident_cellsILb(\d)ELb(\d)E(?:Lb(\d)E)?', f[:300])
        if m and (m.group(1), m.group(2), m.group(3) or '0') == (mlp, kc, v2):
            return [re.sub(r'/\* 0x[0-9a-f]+ \*/', '', l).strip() for l in f.splitlines() if re.search(r'/\*[0-9a-f]{4}\*/', l)]
if __name__ == '__main__':
    new = sys.argv[1]
    for k in (('1','1','0'), ('1','0','0'), ('0','0','0')):
        print(k, 'identical to HEAD:', fn('head.sass', *k) == fn(new, *k))
    v = fn(new, '1', '1', '1'); print('v2 instr', len(v), 'LDL/STL', sum(('LDL' in l or 'STL' in l) for l in v))

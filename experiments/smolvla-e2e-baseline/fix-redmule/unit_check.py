"""Check the stdout of unit_test_main.c: python3 -I unit_check.py <out.bin> <trunc|rne>"""
import sys
import numpy as np

raw = open(sys.argv[1], 'rb').read()
mode = sys.argv[2]
dec = np.frombuffer(raw[:65536 * 4], dtype=np.float32)
ref = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
nan = np.isnan(ref)
assert (np.isnan(dec) == nan).all(), "NaN pattern mismatch in fp16_to_float"
assert (dec[~nan].view(np.uint32) == ref[~nan].view(np.uint32)).all(), "fp16_to_float mismatch"
print("fp16_to_float: all 65536 patterns exact (subnormals, +-0, inf, NaN)")

rec = np.frombuffer(raw[65536 * 4:], dtype=np.dtype([('b', '<u4'), ('r', '<u2'), ('p', '<u2')]))
b = rec['b']
r = rec['r']
f = b.view(np.float32)
with np.errstate(all='ignore'):
    n = f.astype(np.float16)                     # numpy: IEEE round-to-nearest-even
    if mode == 'rne':
        exp = n.view(np.uint16)
    else:                                        # round toward zero, overflow (>= 65536) -> inf
        nb = n.view(np.uint16).copy()
        back = (~np.isnan(n)) & (np.abs(n.astype(np.float64)) > np.abs(f.astype(np.float64)))
        nb[back] -= 1
        big = np.abs(f) >= 65536
        nb[big] = ((b[big] >> 16) & 0x8000).astype(np.uint16) | 0x7C00
        exp = nb
isn = np.isnan(f)
bad = (~isn) & (r != exp)
badn = isn & (((r & 0x7C00) != 0x7C00) | ((r & 0x3FF) == 0))
print(f"float_to_fp16 [{mode}]: {len(f)} cases, {int(bad.sum())} mismatches, {int(badn.sum())} NaN failures")
for i in np.nonzero(bad)[0][:10]:
    print("  bad", hex(b[i]), "got", hex(r[i]), "expected", hex(exp[i]))
names = {0x0: '+0', 0x80000000: '-0', 1: 'f32 denormal', 0x7F800000: '+inf', 0xFF800000: '-inf', 0x7FC00000: 'NaN',
         0x7F800001: 'sNaN', 0x33800000: '2^-24 (min sub)', 0x33000000: '2^-25 (tie)', 0x33000001: '2^-25+ulp',
         0x337FFFFF: '~2^-24 -ulp', 0x387FC000: 'max sub', 0x38800000: 'min normal', 0x477FE000: '65504',
         0x477FF000: '65520', 0x47800000: '65536', 0x34000000: '2^-23'}
for i in range(len(f) - 21, len(f)):
    if int(b[i]) in names:
        print(f"  edge {names[int(b[i])]:16s} {hex(b[i])} -> {hex(r[i])}")
sys.exit(1 if bad.any() or badn.any() else 0)

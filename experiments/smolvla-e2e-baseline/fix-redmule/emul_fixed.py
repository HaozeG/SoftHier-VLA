"""Q[23,273] kk=3 K-tile of the first sh_gemm (diag8), replayed with old / fixed (A, truncating) / fixed+RNE (B) logic.
usage: python emul_fixed.py   (needs numpy; reads ../diag8/peek_output.txt)"""
import os
import re
import struct
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
X, W = [], []
for ln in open(os.path.join(here, '..', 'diag8', 'peek_output.txt')):
    m = re.match(r'\[peek([XW])\] (\d+) (.*)', ln)
    if m:
        (X if m.group(1) == 'X' else W).extend(int(t, 16) for t in m.group(3).split())
assert len(X) == 128 and len(W) == 128, (len(X), len(W))
bits = lambda v: struct.unpack('I', struct.pack('f', v))[0]
fl = lambda b: struct.unpack('f', struct.pack('I', b))[0]


def f2h_old(v):
    b = bits(v); s = (b >> 31) << 15; ex = ((b >> 23) & 255) - 127 + 15; man = (b & 0x7fffff) >> 13
    if ex <= 0:
        if ex < -10: return s
        return (s | (((b & 0x7fffff) | 0x800000) >> (1 - ex))) & 0xffff
    if ex >= 31: return s | 0x7c00
    return s | (ex << 10) | man


def f2h_a(v):
    b = bits(v); s = (b >> 31) << 15; ex = ((b >> 23) & 255) - 127 + 15; man = (b & 0x7fffff) >> 13
    if ((b >> 23) & 255) == 255: return s | 0x7c00 | (0x200 if b & 0x7fffff else 0)
    if ex <= 0:
        if ex < -10: return s
        return s | ((((b & 0x7fffff) | 0x800000) >> (14 - ex)) & 0x3ff)
    if ex >= 31: return s | 0x7c00
    return s | (ex << 10) | man


def f2h_b(v):
    b = bits(v); s = (b >> 31) << 15; ex = ((b >> 23) & 255) - 127 + 15; m23 = b & 0x7fffff
    if ((b >> 23) & 255) == 255: return s | 0x7c00 | (0x200 if m23 else 0)
    if ex >= 31: return s | 0x7c00
    if ex <= 0:
        if ex < -10: return s
        sig = m23 | 0x800000; sh = 14 - ex; q = sig >> sh; rem = sig & ((1 << sh) - 1); half = 1 << (sh - 1)
        if rem > half or (rem == half and (q & 1)): q += 1
        return s | q
    q = (ex << 10) | (m23 >> 13); rem = m23 & 0x1fff
    if rem > 0x1000 or (rem == 0x1000 and (q & 1)): q += 1
    return s | q


def h2f_old(h):
    e = (h >> 10) & 31
    return fl(((h >> 15) << 31) | ((0 if e == 0 else e + 112) << 23) | ((h & 0x3ff) << 13))


def h2f_new(h):
    e = (h >> 10) & 31; m = h & 0x3ff
    if e == 0: return (-1.0 if h >> 15 else 1.0) * m * 2.0 ** -24
    return fl(((h >> 15) << 31) | ((255 if e == 31 else e + 112) << 23) | (m << 13))


def run(f2h, h2f):
    acc = np.float32(h2f(0xb21d))
    for a, b in zip(X, W):
        acc = np.float32(np.float32(h2f(a)) * np.float32(h2f(b)) + acc)
    return f2h(float(acc))


print('old  :', hex(run(f2h_old, h2f_old)), '(observed in GVSoC: 0xc000)')
ra = run(f2h_a, h2f_new); rb = run(f2h_b, h2f_new)
print('A    :', hex(ra), float(np.array(ra, dtype=np.uint16).view(np.float16)))
print('A+B  :', hex(rb), float(np.array(rb, dtype=np.uint16).view(np.float16)))

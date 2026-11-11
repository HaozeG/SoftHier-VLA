import re, struct, numpy as np
X=[];W=[]
for ln in open('/home/haoze/SoftHier-VLA/.claude/worktrees/smolvla-e2e-baseline/experiments/smolvla-e2e-baseline/softhier-mlir/tests/gvsoc/mlir_app/last_run.log'):
    m=re.match(r'\[peek([XW])\] (\d+) (.*)',ln)
    if m:
        v=[int(t,16) for t in m.group(3).split()]
        (X if m.group(1)=='X' else W).extend(v)
X=np.array(X,dtype=np.uint16); W=np.array(W,dtype=np.uint16)
f=lambda a: a.view(np.float16).astype(np.float32)
x=f(X); w=f(W)
y0=np.float32(np.array(0xb21d,dtype=np.uint16).view(np.float16))
print("y0",y0)
# model: acc=fp16->float(y); acc=(fa*fb)+acc sequential float32
def fp16_to_float(h):
    s=(h>>15)&1; e=(h>>10)&31; m=h&0x3ff
    eb=0 if e==0 else e+127-15
    return struct.unpack('f',struct.pack('I',(s<<31)|(eb<<23)|(m<<13)))[0]
def float_to_fp16(v):
    b=struct.unpack('I',struct.pack('f',v))[0]
    sign=(b>>31)<<15; ex=((b>>23)&255)-127+15; man=(b&0x7fffff)>>13
    if ex<=0:
        if ex<-10: return sign
        man=((b&0x7fffff)|0x800000)>>(1-ex); return (sign|man)&0xffff
    elif ex>=31: return sign|0x7c00
    return sign|(ex<<10)|man
acc=np.float32(fp16_to_float(0xb21d))
for a,b in zip(X,W):
    acc=np.float32(np.float32(fp16_to_float(int(a)))*np.float32(fp16_to_float(int(b)))+acc)
r=float_to_fp16(float(acc)); print("model acc",acc,"fp16 bits",hex(r),float(np.array(r,dtype=np.uint16).view(np.float16)))
# also double precision exact
ex=float(np.array(0xb21d,dtype=np.uint16).view(np.float16))+sum(float(a)*float(b) for a,b in zip(x,w))
print("exact",ex)
# running sums
acc=np.float32(fp16_to_float(0xb21d)); mn=1e9
for i,(a,b) in enumerate(zip(X,W)):
    acc=np.float32(np.float32(fp16_to_float(int(a)))*np.float32(fp16_to_float(int(b)))+acc)
    if abs(acc)<1e-3: print("small intermediate at k",i,acc)
# proposed GVSoC fix: subnormal shift 14-ex instead of 1-ex
def float_to_fp16_fixed(v):
    b=struct.unpack('I',struct.pack('f',v))[0]
    sign=(b>>31)<<15; ex=((b>>23)&255)-127+15; man=(b&0x7fffff)>>13
    if ex<=0:
        if ex<-10: return sign
        return sign|(((b&0x7fffff)|0x800000)>>(14-ex))
    elif ex>=31: return sign|0x7c00
    return sign|(ex<<10)|man
rf=float_to_fp16_fixed(float(acc)); print("fixed fp16 bits",hex(rf),float(np.array(rf,dtype=np.uint16).view(np.float16)),"vs exact",ex)
print("acc fp32 bits", hex(struct.unpack('I',struct.pack('f',float(acc)))[0]))

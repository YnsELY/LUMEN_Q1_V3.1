"""Actual CPU byte streams: rANS + mixed-radix shell ranks + explicit escapes.

This is a correctness oracle, not the LUMEN Slot-CP format, and not a CUDA
kernel. One rANS lane per stream, tables local to a stream, no hidden estimate
of padding. The outer checksum detects accidental corruption, not forgery.
"""
from __future__ import annotations
from dataclasses import dataclass
import bisect
import hashlib
import math
import numbers
import struct
from typing import Sequence
import numpy as np
from . import e8

def _int(value, name, minimum=0, maximum=None):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Integral):
        raise ValueError(name+" must be an integer")
    value=int(value)
    if value<minimum or (maximum is not None and value>maximum):
        raise ValueError(name+" outside supported range")
    return value


RANS_L=1<<23
MAGIC=b"LE8CPU3\x00"
HEADER=struct.Struct("<8sBBBBIQQQQ")  # mode, shell_limit, precision, reserved, group_size, n, lengths


def normalized_frequencies(counts:Sequence[int], precision:int=14)->list[int]:
    """Positive integer frequencies; deterministic largest-remainder rounding."""
    precision=_int(precision,"precision",1,16)
    counts=[_int(c,"count") for c in counts]
    n=len(counts); total=1<<precision
    if not 1<=precision<=16 or not 1<=n<=total or any(c<0 for c in counts):
        raise ValueError("Invalid frequency counts")
    weight=[int(c)+1 for c in counts]  # explicit Laplace prior, unlike v2's false comment
    den=sum(weight); remaining=total-n
    f=[1+(remaining*w)//den for w in weight]
    order=sorted(range(n),key=lambda i:(-(remaining*weight[i]%den),i))
    for i in order[:total-sum(f)]: f[i]+=1
    return f


def _cdf(freq):
    if len(freq)==0: raise ValueError("Empty frequencies")
    freq=[_int(f,"frequency",1,65536) for f in freq]
    out=[0]
    for f in freq: out.append(out[-1]+int(f))
    total=out[-1]
    if total<2 or total&(total-1) or total>65536: raise ValueError("Frequency total must be a supported power of 2")
    return out, total.bit_length()-1


def rans_encode(symbols:Sequence[int],freq:Sequence[int])->bytes:
    cdf,bits=_cdf(freq); x=RANS_L; emitted=bytearray()
    for s in reversed(symbols):
        s=_int(s,"symbol")
        if not 0<=s<len(freq): raise ValueError("Symbol out of range")
        f=int(freq[s]); xmax=((RANS_L>>bits)<<8)*f
        while x>=xmax:
            emitted.append(x&255); x>>=8
        x=((x//f)<<bits)+(x%f)+cdf[s]
    return struct.pack("<I",x)+bytes(reversed(emitted))


def rans_decode(payload:bytes,freq:Sequence[int],count:int)->list[int]:
    cdf,bits=_cdf(freq)
    count=_int(count,"symbol count")
    if len(payload)<4 or count<0: raise ValueError("Truncated rANS stream")
    x=struct.unpack_from("<I",payload)[0]; pos=4; out=[]; mask=(1<<bits)-1
    if not RANS_L<=x<RANS_L*256: raise ValueError("Invalid rANS initial state")
    for _ in range(count):
        slot=x&mask; s=bisect.bisect_right(cdf,slot)-1
        out.append(s); x=int(freq[s])*(x>>bits)+slot-cdf[s]
        while x<RANS_L:
            if pos>=len(payload): raise ValueError("Truncated rANS renormalization")
            x=(x<<8)|payload[pos]; pos+=1
    if x!=RANS_L or pos!=len(payload):
        raise ValueError("Noncanonical rANS termination or trailing data")
    return out


def _pack_ranks(ranks,radices,group_size):
    data=bytearray()
    for start in range(0,len(ranks),group_size):
        v=0; p=1
        for r,m in zip(ranks[start:start+group_size],radices[start:start+group_size]):
            if not 0<=r<m: raise ValueError("Rank out of range")
            v=v*m+r; p*=m
        nbytes=((p-1).bit_length()+7)//8
        data.extend(v.to_bytes(nbytes,'big'))
    return bytes(data)


def _unpack_group_tree(v:int,rr:Sequence[int])->list[int]:
    if len(rr)==1: return [v]
    mid=len(rr)//2; rp=math.prod(rr[mid:]); q,r=divmod(v,rp)
    return _unpack_group_tree(q,rr[:mid])+_unpack_group_tree(r,rr[mid:])


def _unpack_ranks(data,radices,group_size):
    out=[]; pos=0
    for start in range(0,len(radices),group_size):
        rr=radices[start:start+group_size]; p=math.prod(rr)
        nbytes=((p-1).bit_length()+7)//8
        if pos+nbytes>len(data): raise ValueError("Truncated rank stream")
        v=int.from_bytes(data[pos:pos+nbytes],'big'); pos+=nbytes
        if v>=p: raise ValueError("Invalid mixed-radix word")
        out.extend(_unpack_group_tree(v,rr) if rr else [])
    if pos!=len(data): raise ValueError("Trailing rank data")
    return out


def _symbols(codes,limit,mode):
    esc=limit*(2 if mode else 1)
    syms=[]
    for u in codes:
        m=e8.shell(u)
        syms.append(esc if m>=limit else (2*m+(int(u[0])&1) if mode else m))
    return syms,esc


@dataclass(frozen=True)
class ShellModel:
    frequencies:tuple[int,...]
    limit:int=64
    mode:int=0  # 0 shell-uniform; 1 shell+coset conditional

    def __post_init__(self):
        _int(self.mode,"mode",0,1); _int(self.limit,"limit",1,128)
        object.__setattr__(self,'frequencies',tuple(_int(f,"frequency",1,65536) for f in self.frequencies))
        if self.mode not in (0,1) or not 1<=self.limit<=128:
            raise ValueError("Invalid shell model")
        _cdf(self.frequencies)
        if len(self.frequencies)!=self.limit*(2 if self.mode else 1)+1:
            raise ValueError("Wrong table length")

    @classmethod
    def fit(cls,codes,limit=64,mode=0,precision=14):
        limit=_int(limit,"limit",1,128);mode=_int(mode,"mode",0,1)
        syms,esc=_symbols(codes,limit,mode)
        c=[0]*(esc+1)
        for s in syms:c[s]+=1
        return cls(tuple(normalized_frequencies(c,precision)),limit,mode)

    @classmethod
    def gaussian(cls,sigma_over_step=1.2,limit=64):
        limit=_int(limit,"limit",1,128)
        if not np.isfinite(sigma_over_step) or sigma_over_step<=0: raise ValueError("Invalid sigma")
        logs=np.array([math.log(e8.shell_size(m))-m/sigma_over_step**2 for m in range(limit)])
        w=np.exp(logs-logs.max())
        counts=[int(v*1_000_000) for v in w]+[1]
        return cls(tuple(normalized_frequencies(counts)),limit,0)

    def bits(self,u)->float:
        """Idealized fixed-table cost, explicitly NOT physical encoded bytes."""
        m=e8.shell(u); esc=self.limit*(2 if self.mode else 1)
        s=esc if m>=self.limit else (2*m+(int(u[0])&1) if self.mode else m)
        cost=-math.log2(self.frequencies[s]/sum(self.frequencies))
        if s==esc:return cost+256.0  # 8 signed int32 coordinates, no saturation
        radix=e8.coset_size(m,int(u[0])&1) if self.mode else e8.shell_size(m)
        return cost+math.log2(radix)

    def total_model_bits(self,codes)->float:
        return sum(self.bits(u) for u in codes)


def encode_e8(codes,model:ShellModel|None=None,group_size:int=128)->bytes:
    codes=[e8.checked(u) for u in codes]
    group_size=_int(group_size,"group_size",1,4096)
    model=ShellModel.fit(codes) if model is None else model
    syms,esc=_symbols(codes,model.limit,model.mode)
    ranks=[]; radices=[]; escaped=bytearray()
    for u,s in zip(codes,syms):
        if s==esc:
            if any(c<-(1<<31) or c>=(1<<31) for c in u):
                raise OverflowError("Escape requires signed int32 coordinates")
            escaped.extend(struct.pack('<8i',*u))
        else:
            m,p,r=e8.rank(u,separate_coset=bool(model.mode))
            ranks.append(r)
            radices.append(e8.coset_size(m,p) if model.mode else e8.shell_size(m))
    a=rans_encode(syms,model.frequencies); b=_pack_ranks(ranks,radices,group_size)
    precision=sum(model.frequencies).bit_length()-1
    # uint32 permits a frequency of 65536 at precision 16; count all table bytes.
    f=struct.pack('<'+'I'*len(model.frequencies),*model.frequencies)
    head=HEADER.pack(MAGIC,model.mode,model.limit,precision,0,group_size,len(codes),len(a),len(b),len(escaped))
    body=head+f+a+b+bytes(escaped)
    return body+hashlib.sha256(body).digest()


def decode_e8(data:bytes,max_vectors:int=1_000_000):
    max_vectors=_int(max_vectors,"max_vectors")
    if not isinstance(data,bytes): raise ValueError("Immutable bytes are required")
    if len(data)<HEADER.size+32: raise ValueError("Truncated E8 container")
    body=data[:-32]
    if hashlib.sha256(body).digest()!=data[-32:]: raise ValueError("Checksum mismatch")
    magic,mode,limit,prec,reserved,group,n,la,lb,le=HEADER.unpack_from(body)
    if magic!=MAGIC or mode not in (0,1) or not 1<=limit<=128 or not 1<=prec<=16 or reserved:
        raise ValueError("Unsupported E8 header")
    if not 1<=group<=4096 or n>max_vectors: raise ValueError("Decode resource limit")
    nf=limit*(2 if mode else 1)+1; pos=HEADER.size; flen=4*nf
    if nf>(1<<prec): raise ValueError("Precision too small for strictly positive frequency table")
    if pos+flen+la+lb+le!=len(body) or le%32:
        raise ValueError("Inconsistent section lengths")
    freq=struct.unpack_from('<'+'I'*nf,body,pos); pos+=flen
    model=ShellModel(tuple(freq),limit,mode)
    if sum(freq)!=(1<<prec): raise ValueError("Frequency precision mismatch")
    syms=rans_decode(body[pos:pos+la],freq,n);pos+=la
    rankbytes=body[pos:pos+lb];pos+=lb; escbytes=body[pos:]
    radices=[]
    for s in syms:
        if s==nf-1:continue
        m=s//2 if mode else s; p=s%2 if mode else None
        size=e8.coset_size(m,p) if mode else e8.shell_size(m)
        if size<=0: raise ValueError("Impossible coset symbol")
        radices.append(size)
    ranks=iter(_unpack_ranks(rankbytes,radices,group)); out=[];ep=0
    for s in syms:
        if s==nf-1:
            if ep+32>len(escbytes):raise ValueError("Missing escape payload")
            u=e8.checked(struct.unpack_from('<8i',escbytes,ep));ep+=32
            if e8.shell(u)<limit:raise ValueError("Noncanonical escape")
        else:
            m=s//2 if mode else s; p=s%2 if mode else None
            u=e8.unrank(m,next(ranks),p)
        out.append(u)
    if ep!=len(escbytes):raise ValueError("Trailing escape payload")
    return np.array(out,dtype=np.int64).reshape(-1,8),model


MATRIX_HEADER=struct.Struct('<8sIIQ')
MATRIX_MAGIC=b'LMATv3\x00\x00'


def encode_matrix(codes,steps,model=None,group_size=128):
    """Store E8 rows and their already rounded FP16 scales, no implicit transform.

    Input codes shape [m,n/8,8]. Additional transforms in a real engine MUST be
    separately serialized and counted. This reference stores canonical bases.
    """
    codes=np.asarray(codes); sf=np.asarray(steps,dtype=np.float64)
    if codes.ndim!=3 or codes.shape[2]!=8 or codes.shape[0]==0 or codes.shape[1]==0:
        raise ValueError("Invalid code shape")
    m,b,_=codes.shape
    if m>=2**32 or b*8>=2**32: raise ValueError('Matrix shape exceeds format limits')
    with np.errstate(over='ignore',under='ignore'):
        st=sf.astype('<f2')
    st64=st.astype(np.float64)
    if sf.shape!=(m,) or not np.all(np.isfinite(sf)) or np.any(sf<=0) or not np.all(np.isfinite(st64)) or np.any(st64<=0) or not np.array_equal(sf,st64):
        raise ValueError("Scales must be exact positive finite FP16 values")
    payload=encode_e8(codes.reshape(-1,8),model,group_size)
    body=MATRIX_HEADER.pack(MATRIX_MAGIC,m,b*8,len(payload))+st.tobytes()+payload
    return body+hashlib.sha256(body).digest()


def decode_matrix(data:bytes,max_weights:int=8_000_000):
    max_weights=_int(max_weights,"max_weights",1)
    if not isinstance(data,bytes): raise ValueError("Immutable bytes are required")
    if len(data)<MATRIX_HEADER.size+32:raise ValueError("Truncated matrix")
    body=data[:-32]
    if hashlib.sha256(body).digest()!=data[-32:]:raise ValueError("Matrix checksum mismatch")
    magic,m,n,length=MATRIX_HEADER.unpack_from(body)
    if magic!=MATRIX_MAGIC or not m or not n or n%8 or m*n>max_weights:
        raise ValueError("Unsupported matrix dimensions")
    pos=MATRIX_HEADER.size
    if pos+2*m+length!=len(body):raise ValueError("Bad matrix lengths")
    st=np.frombuffer(body,dtype='<f2',count=m,offset=pos).astype(np.float64);pos+=2*m
    if not np.all(np.isfinite(st)) or np.any(st<=0):raise ValueError("Invalid scales")
    u,model=decode_e8(body[pos:],max_vectors=max_weights//8)
    if len(u)!=m*n//8:raise ValueError("Code count mismatch")
    u=u.reshape(m,n//8,8)
    q=(u.reshape(m,n)/2)*st[:,None]
    return q,u,st,model


def matrix_codec_metadata(data: bytes, max_weights=8_000_000):
    """Validate the full LMAT artifact before exposing binary codec settings."""
    weights, codes, steps, model = decode_matrix(data, max_weights=max_weights)
    payload_offset=MATRIX_HEADER.size+2*weights.shape[0]
    fields=HEADER.unpack_from(data,payload_offset)
    return {'group_size':int(fields[5]), 'precision':int(fields[3]),
            'mode':model.mode,'limit':model.limit}

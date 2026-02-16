package transformer

import (
	"math"

	"github.com/brucetruth/minigpt/llm/nn"
	"github.com/brucetruth/minigpt/llm/tensor"
)

type CausalSelfAttention struct {
	CAttn *nn.Linear
	CProj *nn.Linear
	
	NHead int
	NEmb  int
	
	// Cache for backward
	input *tensor.NDArray // [B, T, C]
	q     *tensor.NDArray // [B, H, T, D]
	k     *tensor.NDArray // [B, H, T, D]
	v     *tensor.NDArray // [B, H, T, D]
	att   *tensor.NDArray // [B, H, T, T] (probs)
}

func NewCausalSelfAttention(cfg Config) *CausalSelfAttention {
	return &CausalSelfAttention{
		CAttn: nn.NewLinear(cfg.NEmb, 3*cfg.NEmb),
		CProj: nn.NewLinear(cfg.NEmb, cfg.NEmb),
		NHead: cfg.NHead,
		NEmb:  cfg.NEmb,
	}
}

func (csa *CausalSelfAttention) Forward(x *tensor.NDArray) *tensor.NDArray {
	// x: [B, T, C]
	csa.input = x
	B, T, C := x.Shape[0], x.Shape[1], x.Shape[2]
	headDim := C / csa.NHead
	
	// 1. QKV projection
	qkv := csa.CAttn.Forward(x) // [B, T, 3C]
	
	// 2. Split and Reshape
	q := tensor.New(B, csa.NHead, T, headDim)
	k := tensor.New(B, csa.NHead, T, headDim)
	v := tensor.New(B, csa.NHead, T, headDim)
	
	strideQKV := 3 * C
	
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			offsetSrc := (b*T + t) * strideQKV
			
			for h := 0; h < csa.NHead; h++ {
				for d := 0; d < headDim; d++ {
					// Q
					q.Data[((b*csa.NHead + h)*T + t)*headDim + d] = qkv.Data[offsetSrc + h*headDim + d]
					// K
					k.Data[((b*csa.NHead + h)*T + t)*headDim + d] = qkv.Data[offsetSrc + C + h*headDim + d]
					// V
					v.Data[((b*csa.NHead + h)*T + t)*headDim + d] = qkv.Data[offsetSrc + 2*C + h*headDim + d]
				}
			}
		}
	}
	
	csa.q = q
	csa.k = k
	csa.v = v
	
	// 3. Q @ K^T
	kt := tensor.Transpose(k)
	att := tensor.MatMul(q, kt) // [B, H, T, T]
	
	// Scale
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for i := range att.Data {
		att.Data[i] *= scale
	}
	
	// Mask (Causal)
	minVal := float32(math.Inf(-1)) // Should handle this appropriately if strict float32
	// For compat: usually -1e9 or similar
	minVal = -1e9
	
	for t1 := 0; t1 < T; t1++ {
		for t2 := 0; t2 < T; t2++ {
			if t2 > t1 {
				for b := 0; b < B*csa.NHead; b++ {
					att.Data[(b*T + t1)*T + t2] = minVal
				}
			}
		}
	}
	
	// Softmax
	probs := tensor.Softmax(att)
	csa.att = probs
	
	// 4. Probs @ V
	y := tensor.MatMul(probs, v)
	
	// 5. Reassemble -> [B, T, C]
	out := tensor.New(B, T, C)
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for h := 0; h < csa.NHead; h++ {
				for d := 0; d < headDim; d++ {
					val := y.Data[((b*csa.NHead + h)*T + t)*headDim + d]
					out.Data[(b*T + t)*C + h*headDim + d] = val
				}
			}
		}
	}
	
	return csa.CProj.Forward(out)
}

func (csa *CausalSelfAttention) Backward(gradOutput *tensor.NDArray) *tensor.NDArray {
	// gradOutput: [B, T, C]
	// 1. dCProj
	dY_flat := csa.CProj.Backward(gradOutput) // [B, T, C]
	
	// Reshape dY_flat to [B, H, T, D]
	B, T := csa.input.Shape[0], csa.input.Shape[1]
	headDim := csa.NHead // Mistake in var name matching
	headDim = csa.NEmb / csa.NHead
	
	dY := tensor.New(B, csa.NHead, T, headDim)
	
	// Inverse reassemble
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for h := 0; h < csa.NHead; h++ {
				for d := 0; d < headDim; d++ {
					val := dY_flat.Data[(b*T + t)*csa.NEmb + h*headDim + d]
					dY.Data[((b*csa.NHead + h)*T + t)*headDim + d] = val
				}
			}
		}
	}
	
	// 2. dV = P^T * dY
	// P: [B, H, T, T]
	pt := tensor.Transpose(csa.att)
	dV := tensor.MatMul(pt, dY) // [B, H, T, D]
	
	// 3. dP = dY * V^T
	vt := tensor.Transpose(csa.v)
	dP := tensor.MatMul(dY, vt) // [B, H, T, T]
	
	// 4. dS = P * (dP - sum(dP * P))
	// Softmax backward
	// dS_ij = P_ij * (dP_ij - sum_k(P_ik * dP_ik))
	
	dS := tensor.New(dP.Shape...)
	
	batch := B * csa.NHead * T // Total rows in P
	stride := T
	
	for i := 0; i < batch; i++ {
		offset := i * stride
		
		var sum float32
		for j := 0; j < stride; j++ {
			pVal := csa.att.Data[offset+j]
			dpVal := dP.Data[offset+j]
			sum += pVal * dpVal
		}
		
		for j := 0; j < stride; j++ {
			pVal := csa.att.Data[offset+j]
			dpVal := dP.Data[offset+j]
			dS.Data[offset+j] = pVal * (dpVal - sum)
		}
	}
	
	// 5. Scale dS
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for i := range dS.Data {
		dS.Data[i] *= scale
	}
	
	// 6. dQ = dS * K  ( [B,H,T,T] * [B,H,T,D] -> [B,H,T,D] )
	dQ := tensor.MatMul(dS, csa.k)
	
	// 7. dK = dS^T * Q ( [B,H,T,T]^T * [B,H,T,D] -> [B,H,T,D] )
	dSt := tensor.Transpose(dS)
	dK := tensor.MatMul(dSt, csa.q)
	
	// 8. Reassemble dQ, dK, dV into dQKV [B, T, 3C]
	dQKV := tensor.New(B, T, 3*csa.NEmb)
	strideQKV := 3 * csa.NEmb
	
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			offsetSrc := (b*T + t) * strideQKV
			
			for h := 0; h < csa.NHead; h++ {
				for d := 0; d < headDim; d++ {
					// Q
					dQKV.Data[offsetSrc + h*headDim + d] = dQ.Data[((b*csa.NHead + h)*T + t)*headDim + d]
					// K
					dQKV.Data[offsetSrc + csa.NEmb + h*headDim + d] = dK.Data[((b*csa.NHead + h)*T + t)*headDim + d]
					// V
					dQKV.Data[offsetSrc + 2*csa.NEmb + h*headDim + d] = dV.Data[((b*csa.NHead + h)*T + t)*headDim + d]
				}
			}
		}
	}
	
	// 9. Back through CAttn
	return csa.CAttn.Backward(dQKV)
}

func (csa *CausalSelfAttention) Parameters() []*nn.Parameter {
	p := csa.CAttn.Parameters()
	p = append(p, csa.CProj.Parameters()...)
	return p
}

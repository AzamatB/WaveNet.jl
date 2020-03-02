using Flux
using Flux: @functor
using NamedTupleTools

@views function gate(x::AbstractArray{<:Real,3})
   split = size(x, 2) ÷ 2
   tanh.(x[:, 1:split, :]) .* σ.(x[:, (split+1):end, :])
end

struct ResBlock{D, R, S, C}
   dilconv  :: D
   resconv  :: R
   skipconv :: S
   condconv :: C
end

struct LastBlock{D, S, C}
   dilconv  :: D
   skipconv :: S
   condconv :: C
end

@functor ResBlock
@functor LastBlock

function ResBlock(dilation::Integer, nch::NamedTuple{(:res, :skip, :cond),<:NTuple{3,Integer}})
   dilconv  = Conv((2,), nch.res => 2nch.res; dilation = dilation, pad = (dilation, 0)) |> gpu
   resconv  = Conv((1,), nch.res => nch.res)  |> gpu
   skipconv = Conv((1,), nch.res => nch.skip) |> gpu
   condconv = Conv((1,), nch.cond => 2nch.res, tanh) |> gpu
   ResBlock(dilconv, resconv, skipconv, condconv)
end

function LastBlock(dilation::Integer, nch::NamedTuple{(:res, :skip, :cond),<:NTuple{3,Integer}})
   dilconv  = Conv((2,), nch.res => 2nch.res; dilation = dilation, pad = (dilation, 0)) |> gpu
   skipconv = Conv((1,), nch.res => nch.skip) |> gpu
   condconv = Conv((1,), nch.cond => 2nch.res, tanh) |> gpu
   LastBlock(dilconv, skipconv, condconv)
end

"""                             residual connection
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃                              ┏━━ tanh ━━┓                       ┃
input ━━┻━━> 2×1 dilatedconv ━━━ + ━━━ ❖          × ━━┳━━ 1×1 resconv ━━━ + ━━━> output
                                 ┃     ┗━━━ σ ━━━━┛   ┃
conditional input ━━━> 1×1 condconv             1×1 skipconv
                                                      ┃
Σskip ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━> + ━━━━━━━━> Σskip
"""
function (m::ResBlock)((Σskip, x)::NTuple{2,DenseArray{<:Real,3}})
   y = gate(m.dilconv(x))
   return (Σskip + m.skipconv(y), x + m.resconv(y))
end

function (m::ResBlock)((Σskip, x, h)::NTuple{3,DenseArray{<:Real,3}})
   y = gate(m.dilconv(x) + m.condconv(h))
   return (Σskip + m.skipconv(y), x + m.resconv(y), h)
end

function (m::LastBlock)((Σskip, x)::NTuple{2,DenseArray{<:Real,3}})
   y = gate(m.dilconv(x))
   return Σskip + m.skipconv(y)
end

function (m::LastBlock)((Σskip, x, h)::NTuple{3,DenseArray{<:Real,3}})
   y = gate(m.dilconv(x) + m.condconv(h))
   return Σskip + m.skipconv(y)
end

struct WaveNet{I, R, O}
   inblock   :: I
   resblocks :: R
   outblock  :: O
end

@functor WaveNet

function WaveNet(nlayers::Integer,
      nch::NamedTuple{(:res, :skip, :cond, :out),<:NTuple{4,Integer}} = (res=64, skip=256, cond=0, out=256); depth::Integer = 10
   )
   nch′ = delete(nch, :out)
   inblock = Conv((2,), 1 => nch.res, tanh; pad=(1,0)) |> gpu

   resblocks = Chain(
      (ResBlock(2^(i % depth), nch′) for i ∈ 0:(nlayers-2))...,
      LastBlock(2^((nlayers-1) % depth), nch′)
   ) |> gpu

   outblock = Chain(
      Conv((1,), nch.skip => nch.out, leakyrelu),
      Conv((1,), nch.out => nch.out),
      softmax
   ) |> gpu
   WaveNet(inblock, resblocks, outblock)
end

function (m::WaveNet)(x::DenseArray{<:Real,3})
   y = m.inblock(x)
   len, _, batch = size(y)
   skip₀ = gpu(zeros(Float32, len, nch.skip, batch))
   Σskip = m.resblocks((skip₀, y))
   return m.outblock(leakyrelu.(Σskip))
end

function (m::WaveNet)(x::DenseArray{<:Real,3}, h::DenseArray{<:Real,3})
   y = m.inblock(x)
   len, _, batch = size(y)
   skip₀ = zeros(Float32, len, nch.skip, batch)
   Σskip = m.resblocks((skip₀, y, h))
   return m.outblock(leakyrelu.(Σskip))
end

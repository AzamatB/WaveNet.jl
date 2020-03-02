using Flux
using Flux: @functor
using NamedTupleTools

@views function gate(x::AbstractArray{<:Real,3})
   split = size(x, 2) ÷ 2
   tanh.(x[:, 1:split, :]) .* σ.(x[:, (split+1):end, :])
end

struct ResBlock{T<:DenseArray{<:Real,3}, D, R, S, C}
   skip     :: Ref{T}
   dilconv  :: D
   resconv  :: R
   skipconv :: S
   condconv :: C
end

struct LastBlock{T<:DenseArray{<:Real,3}, D, S, C}
   skip     :: Ref{T}
   dilconv  :: D
   skipconv :: S
   condconv :: C
end

@functor ResBlock
@functor LastBlock

function ResBlock(dilation::Integer, nch::NamedTuple{(:res, :skip, :cond),<:NTuple{3,Integer}})
   skip = Ref(gpu(zeros(Float32, 0, 0, 0)))
   dilconv  = Conv((2,), nch.res => 2nch.res; dilation = dilation, pad = (dilation, 0)) |> gpu
   resconv  = Conv((1,), nch.res => nch.res)  |> gpu
   skipconv = Conv((1,), nch.res => nch.skip) |> gpu
   condconv = Conv((1,), nch.cond => 2nch.res, tanh) |> gpu
   ResBlock(skip, dilconv, resconv, skipconv, condconv)
end

function LastBlock(dilation::Integer, nch::NamedTuple{(:res, :skip, :cond),<:NTuple{3,Integer}})
   skip = Ref(gpu(zeros(Float32, 0, 0, 0)))
   dilconv  = Conv((2,), nch.res => 2nch.res; dilation = dilation, pad = (dilation, 0)) |> gpu
   skipconv = Conv((1,), nch.res => nch.skip) |> gpu
   condconv = Conv((1,), nch.cond => 2nch.res, tanh) |> gpu
   LastBlock(skip, dilconv, skipconv, condconv)
end

"""                                  residual
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃                              ┏━━ tanh ━━┓                       ┃
input ━━┻━━> 2×1 dilatedconv ━━━ + ━━━ ❖          × ━━┳━━ 1×1 resconv ━━━ + ━━━> output
                                 ┃     ┗━━━ σ ━━━━┛   ┃
conditional input ━━━> 1×1 condconv             1×1 skipconv
                                                      ┃
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━> + ━━━━━━━━> skip
"""
function (m::ResBlock)(x::DenseArray{<:Real,3})
   y = gate(m.dilconv(x))
   m.skip[] = m.skipconv(y)
   return x + m.resconv(y)
end

function (m::ResBlock)((x, h)::NTuple{2,DenseArray{<:Real,3}})
   y = gate(m.dilconv(x) + m.condconv(h))
   m.skip[] = m.skipconv(y)
   return (x + m.resconv(y), h)
end

function (m::LastBlock)(x::DenseArray{<:Real,3})
   y = gate(m.dilconv(x))
   m.skip[] = m.skipconv(y)
end

function (m::LastBlock)((x, h)::NTuple{2,DenseArray{<:Real,3}})
   y = gate(m.dilconv(x) + m.condconv(h))
   m.skip[] = m.skipconv(y)
end

skip(m::ResBlock)  = m.skip[]
skip(m::LastBlock) = m.skip[]

struct WaveNet{I, R, O}
   inputnet  :: I
   resnet!   :: R
   outputnet :: O
end

@functor WaveNet

function WaveNet(nlayers::Integer,
      nch::NamedTuple{(:res, :skip, :cond, :out),<:NTuple{4,Integer}} = (res=64, skip=256, cond=0, out=256); depth::Integer = 10
   )
   nch′ = delete(nch, :out)
   inputnet = Conv((2,), 1 => nch.res, tanh; pad=(1,0)) |> gpu

   resnet! = Chain(
      (ResBlock(2^(i % depth), nch′) for i ∈ 0:(nlayers-2))...,
      LastBlock(2^((nlayers-1) % depth), nch′)
   ) |> gpu

   outputnet = Chain(
      Conv((1,), nch.skip => nch.out, leakyrelu),
      Conv((1,), nch.out => nch.out),
      softmax
   ) |> gpu
   WaveNet(inputnet, resnet!, outputnet)
end

function (m::WaveNet)(x::DenseArray{<:Real,3}, h::DenseArray{<:Real,3})
   m.resnet!((m.inputnet(x), h))
   m.outputnet(leakyrelu.(sum(skip, m.resnet!)))
end

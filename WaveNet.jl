using Flux
using Flux: @functor

@views function gate(X::DenseArray{<:Real,3})
   split = size(X, 2) ÷ 2
   tanh.(X[:, 1:split, :]) .* σ.(X[:, split+1:end, :])
end

mutable struct Skipped{T <: DenseArray{<:Real,3}}
   output::T
end

@functor Skipped

Skipped() = Skipped(gpu(zeros(Float32, 0, 0, 0)))

struct ResBlock{T<:DenseArray{<:Real,3}, D, R, S, C}
   skip     :: Skipped{T}
   dilconv  :: D
   resconv  :: R
   skipconv :: S
   condconv :: C
end

@functor ResBlock

function ResBlock(dilation::Integer, nch::NamedTuple{(:res,:skip,:cond),<:NTuple{3,Integer}})
   dilconv  = Conv((2,), nch.res => 2nch.res; dilation = dilation, pad = (dilation, 0)) |> gpu
   resconv  = Conv((1,), nch.res => nch.res)  |> gpu
   skipconv = Conv((1,), nch.res => nch.skip) |> gpu
   condconv = Conv((1,), nch.cond => 2nch.res, tanh) |> gpu
   ResBlock(Skipped(), dilconv, resconv, skipconv, condconv)
end

nch = (res=3, skip=5, cond=7)
dilation = 4
m = ResBlock(dilation, nch)
X = rand(Float32, 17, 3, 11)

function (m::ResBlock)(X::DenseArray{<:Real,3})
   Y = gate(m.dilconv(X))
   m.skip.output = m.skipconv(Y)
   X + m.resconv(Y)
end

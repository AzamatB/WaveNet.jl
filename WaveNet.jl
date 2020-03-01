using Flux
using Flux: @functor

@views function gate(X::DenseArray{<:Real,3})
   split = size(X, 2) ÷ 2
   tanh.(X[:, 1:split, :]) .* σ.(X[:, (split+1):end, :])
end

struct ResBlock{T<:DenseArray{<:Real,3}, D, R, S, C}
   skip     :: Ref{T}
   dilconv  :: D
   resconv  :: R
   skipconv :: S
   condconv :: C
end

@functor ResBlock

function ResBlock(dilation::Integer, nch::NamedTuple{(:res,:skip,:cond),<:NTuple{3,Integer}})
   skip = Ref(gpu(zeros(Float32, 0, 0, 0)))
   dilconv  = Conv((2,), nch.res => 2nch.res; dilation = dilation, pad = (dilation, 0)) |> gpu
   resconv  = Conv((1,), nch.res => nch.res)  |> gpu
   skipconv = Conv((1,), nch.res => nch.skip) |> gpu
   condconv = Conv((1,), nch.cond => 2nch.res, tanh) |> gpu
   ResBlock(skip, dilconv, resconv, skipconv, condconv)
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
function (m::ResBlock)(X::DenseArray{<:Real,3})
   Y = gate(m.dilconv(X))
   m.skip[] = m.skipconv(Y)
   return X + m.resconv(Y)
end

function (m::ResBlock)((X, H)::NTuple{2,DenseArray{<:Real,3}})
   Y = gate(m.dilconv(X) + m.condconv(H))
   m.skip[] = m.skipconv(Y)
   return (X + m.resconv(Y), H)
end

function (last_layer::ResBlock)((X, H)::NTuple{2,DenseArray{<:Real,3}})
   Y = gate(m.dilconv(X) + m.condconv(H))
   m.skip[] = m.skipconv(Y)
end

function (m::WaveNet)(X::DenseArray{<:Real,3}, H::DenseArray{<:Real,3})
m.res_blocks!((m.conv1(X), H))


nch = (res=3, skip=5, cond=7)
dilation = 4
m = ResBlock(dilation, nch)
X = rand(Float32, 17, 3, 11)

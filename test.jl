nch = (res=3, skip=5, cond=7)
dilation = 4
m = ResBlock(dilation, nch)
θ = params(m)
len, batch = 17, 11
x = rand(Float32, len, nch.res, batch)
h = rand(Float32, len, nch.cond, batch)
skip₀ = zeros(Float32, len, nch.skip, batch)

m((skip₀, x))
m((skip₀, x, h))

gradient(θ) do
   sum(m((skip₀, x))[1]) + sum(m((skip₀, x))[2])
end
gradient(θ) do
   sum(m((skip₀, x, h))[1]) + sum(m((skip₀, x, h))[2]) + sum(m((skip₀, x, h))[3])
end


nch = (res=3, skip=5, cond=7, out=11)
m = WaveNet(13, nch)
θ = params(m)

x = rand(Float32, len, 1, batch)
h = rand(Float32, len, nch.cond, batch)

m(x)
m(x, h)
gradient(θ) do
   sum(m(x))
end
gradient(θ) do
   sum(m(x, h))
end

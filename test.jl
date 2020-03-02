nch = (res=3, skip=5, cond=7)
dilation = 4
m = ResBlock(dilation, nch)
θ = params(m)
X = rand(Float32, 17, 3, 11)
H = rand(Float32, 17, 7, 11)
XH = (X, H)

m(X)
m(XH)

gradient(θ) do
    sum(m(X) .* 1)
end
gradient(θ) do
    sum(m(XH)[1] .* 1) + sum(m(XH)[2] .* 1)
end

using Flux, FillArrays, IterTools, ControlSystems, Printf
using Flux: params, train!, data, throttle

Gc = tf(1,[1,0.1,1])
G = c2d(ss(Gc), 1)[1]
A,B,C = G.A, G.B, G.C
N = 100

function gen()
    x = [randn(2)]
    u = [randn(1)]
    for i = 1:N-1
        push!(x, A*x[i] + B*u[i] + 0.0randn(2))
        push!(u, randn(1))
    end
    y = Ref(C) .* x
    x,u,y
end

x,u,y = gen()
X = reduce(hcat, x);
U = reduce(hcat, u);
Y = reduce(hcat, y);
# w = (x[:,1:end-1]/x[:,2:end])'
input = vcat.(y,u)
INPUT = [Y;U]


##
untrack(x) =  Flux.mapleaves(data,x)
data1 = (input[1:end-1], y[2:end])
dataset = Iterators.repeated(data1, 100)

f = RNN(2,2) |> untrack #Chain(Dense(2,2,tanh), Dense(2,2))
g = Chain(Dense(2,2,tanh), Dense(2,1)) |> untrack
h = Chain(Dense(2,2,tanh), Dense(2,2)) |> untrack
model = Chain(h...,f,g...)

model.(input)
grads = Zygote.gradient(m->sum(abs2, m(input[1])), model)

pars = []
Flux.mapleaves(model) do p
    p isa AbstractArray || return
    push!(pars, p)
end

function loss()
    l = mean(norm, model(INPUT)-Y)
    l
end
Zygote.refresh()
loss()
opt = [ADAM(0.05) for i in eachindex(pars)]

cb = function()

    @printf("Loss: %10.3f  Linearity: %10.3f\n", loss(), linearityloss())
    # xv = LinRange(extrema(X)..., 20)
    # yh = data.(model.(zip(x,u)))
    # @show mean(norm, yh[1:end-1] .- y[2:end])
    # surface(xv,xv,(x,y)->data(model(([x,y],0)))[1], subplot=1, layout=2)
    # surface!(xv,xv,(x,y)->data(model(([x,y],0)))[2], subplot=2)
    # plot3d!(X[:,1],X[:,2], first.(y), subplot=1, m=(:cyan, 3))
    # plot3d!(X[:,1],X[:,2], last.(y), subplot=2, m=(:cyan, 3))
    # plot3d!(X[1:end-1,1],X[1:end-1,2], X[2:end,1], subplot=1, m=(:red, 3))
    # plot3d!(X[1:end-1,1],X[1:end-1,2], X[2:end,2], subplot=2, m=(:red, 3)) |> display
    plot(Y[2:end], label="y")
    plot!(model(INPUT)', lab="pred") |> display
end

cb()
function train2(loss, pars, dataset, opt; cb=()->nothing)
    # tcb = throttle(cb, 3)
    ps = Params(pars)
    for d in dataset
        Flux.reset!(model)
        grads = Zygote.gradient(()->loss(), ps)
        for i in eachindex(opt)
            g = grads[pars[i]]
            g === nothing && continue
            Flux.Optimise.update!(opt[i], pars[i], g)
        end
        cb()
    end
end
train2(loss, pars, dataset, opt, cb=cb)
# train!(loss, ps, dataset, opt, cb=throttle(cb, 2))

##
function linearest()
    Y = f.state[:,2:end]'
    A = f.state[:,1:end-1]'
    w = A\Zygote.dropgrad(Y)
end

function linearityloss()
    Y = f.state[:,2:end]'
    A = f.state[:,1:end-1]'
    w = A\Zygote.dropgrad(Y)
    lr = mean(norm, Y-A*w)
end

function lossr()
    l = loss()
    lr = linearityloss()
    # lr *= Zygote.dropgrad(l)/Zygote.dropgrad(lr) # Normalize but do not propagate gradients
    l + lr
end
Zygote.refresh()
lossr()
linearityloss()

opt = [ADAM(0.01) for i in eachindex(pars)]
train2(lossr, pars, dataset, opt, cb=cb)

linearest()






loss(data1...)
grads = Zygote.gradient(()->loss(data1...), ps)
Flux.Optimise.update!(opt, ps, grads)

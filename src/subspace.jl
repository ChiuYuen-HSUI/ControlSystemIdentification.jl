
"""
    N4SIDStateSpace is the result of statespace model estimation using the `n4sid` method.

# Fields:
- `sys`: the estimated model in the form of a [`StateSpace`](@ref) object
- `Q`: the estimated covariance matrix of the states
- `R`: the estimated covariance matrix of the measurements
- `S`: the estimated cross covariance matrix between states and measurements
- `K`: the kalman observer gain
- `P`: the solution to the Riccatti equation
- `x`: the estimated state trajectory
- `s`: The singular values
- `fve`: Fraction of variance explained by singular values
"""
struct N4SIDStateSpace <: AbstractStateSpace
    sys
    Q
    R
    S
    K
    P
	x
    s
    fve
end

@inline function Base.getproperty(res::N4SIDStateSpace, p::Symbol)
	if p ∈ (:A, :B, :C, :D, :nx, :ny, :nu, :Ts)
		return getproperty(res.sys, p)
	end
	return getfield(res,p)
end

proj(A,B) = A*B'/(B*B')

@static if VERSION < v"1.3"
    (LinearAlgebra.I)(n) = Matrix{Float64}(I,n,n)
end

"""
    res = n4sid(data, r=:auto; verbose=false)

Estimate a statespace model using the n4sid method. Returns an object of type [`N4SIDStateSpace`](@ref) where the model is accessed as `res.sys`.

#Arguments:
- `data`: Identification data `data = iddata(y,u)`
- `y`: Measurements N×ny
- `u`: Control signal N×nu
- `r`: Rank of the model (model order)
- `verbose`: Print stuff?
- `i`: Algorithm parameter, generally no need to tune this
- `γ`: Set this to a value between (0,1) to stabilize unstable models such that the largest eigenvalue has magnitude γ.
"""
function n4sid(data::InputOutputData,r = :auto;
                    verbose=false,
                    i = r === :auto ? min(length(data)÷20,20) : r+10,
                    γ = nothing,
                    svd = svd,
                    estimator = \)

    y, u = time1(output(data)), time1(input(data))
    N, l = size(y,1),size(y,2)
    m = size(u, 2)
    j = N - 2i

    function hankel(u::AbstractArray,i1,i2)
        d = size(u,2)
        w = (i2-i1+1)
        H = zeros(eltype(u), w*d, j)
        for r = 1:w, c = 1:j
            H[(r-1)*d+1:r*d,c] = u[i1+r+c-1,:]
        end
        H
    end
    mi    = m*i
    li    = l*i
    U0im1 = hankel(u,0,i-1)
    Y0im1 = hankel(y,0,i-1)
    Y0i   = hankel(y,0,i)
    U0i   = hankel(u,0,i)
    UY0   = [U0im1; hankel(u,i,2i-1); Y0im1]
    UY1   = [U0i; hankel(u,i+1,2i-1); Y0i]
    Li    = proj(hankel(y,i,2i-1), UY0)
    Lip1  = proj(hankel(y,i+1,2i-1), UY1)

    L¹ᵢ = Li[:,1:mi]
    L³ᵢ = Li[:,end-li+1:end]

    L¹ᵢp1 = Lip1[:,1:m*(i+1)]
    L³ᵢp1 = Lip1[:,end-l*(i+1)+1:end]

    # Zi = Li*UY0
    # Zip1 = Lip1*UY1

    Zi = [L¹ᵢ L³ᵢ] * [U0im1; Y0im1]

    s = svd(Zi)
    if r === :auto
        r = sum(s.S .> sqrt(s.S[1]*s.S[end]))
        verbose && @info "Choosing order $r"
    end
    n = r
    U1 = s.U[:,1:r]
    S1 = s.S[1:r]
    fve = sum(S1)/sum(s.S)
    verbose && @info "Fraction of variance explained: $(fve)"

    Γi = U1 * Diagonal(sqrt.(S1))
    Γim1 = U1[1:end-l,:] * Diagonal(sqrt.(S1))
    Xi = estimator(Γi,  Zi)
    Xip1 = estimator(Γim1, [L¹ᵢp1 L³ᵢp1] * [U0i; Y0i])

    XY = [Xip1 ; hankel(y,i,i)]
    XU = [Xi ; hankel(u,i,i)]
    L = estimator(XU', XY')'

    A = L[1:n,1:n]
    if γ !== nothing
        e = maximum(abs, eigvals(A))
        if e > γ
            verbose && @info "Stabilizing A, maximum eigval had magnitude $e"
            L = stabilize(L, XU, i, j, m, n, γ)
            A = L[1:n,1:n]
        end
    end
    B = L[1:n,n+1:end]
    C = L[n+1:end,1:n]
    D = L[n+1:end,n+1:end]


    errors = XY - L*XU
    Σ = 1/(j-(n+1))*errors*errors'
    Q = Symmetric(Σ[1:n, 1:n])
    R = Symmetric(Σ[n+1:end, n+1:end])
    S = Σ[1:n, n+1:end]

    local P,K
    try
        P = Symmetric(dare(copy(A'), copy(C'), (Q), (R))) # TODO: Skipped S as dare does not support it
        K = ((C*P*C' + R)\(A*P*C' + S)')'
    catch
        P = fill(NaN, n, n)
        K = fill(NaN, n, l)
    end

    sys = ss(A,B,C,D,sampletime(data))

    N4SIDStateSpace(sys, Q, R, S, K, P, Xi, s.S, fve)
end



"""
"Imposing Stability in Subspace Identification by Regularization", Gestel, Suykens, Dooren, Moor
"""
function stabilize(L, XU, i, j, m, n, γ)
    UtXt = XU[[n+1:end; 1:n],:]'

    F   = qr(UtXt)
    R22 = F.R[m+1:end, m+1:end]
    Σ   = R22'R22
    P2  = -γ^2 * I(n^2)
    P1  = -γ^2 * kron(I(n), Σ) - γ^2 * kron(Σ, I(n))
    A   = L[1:n,1:n]
    AΣ  = A*Σ
    P0  = kron(AΣ, AΣ) - γ^2 * kron(Σ, Σ)

    θ = eigen(Matrix([0I(n^2) -I(n^2); P0 P1]), -Matrix([I(n^2) 0I(n^2); 0I(n^2) P2])).values
    c = maximum(abs.(θ[(imag.(θ) .== 0 ) .* (real.(θ) .> 0)]))

    Σ_XU = XU*XU'
    mod = Σ_XU/(Σ_XU + c*diagm(0=>[ones(n); zeros(m)]))#[I(n) zeros(n,m); zeros(m, n+m)])
    L[1:n,:] .= L[1:n,:]*mod
    L
end


function sysfilter!(state::AbstractVector, res::N4SIDStateSpace, y, u)
	sys = res.sys
	@unpack A,B,C,D,ny = sys
	@unpack K = res
	yh     = vec(C*state + D*u)
	e      = y - yh
	state .= vec(A*state + B*u + K*e)
	yh
end

SysFilter(res::N4SIDStateSpace,x0=zeros(sys.nx)) = SysFilter(res,x0,zeros(eltype(x0), res.sys.ny))

function simulate(res::N4SIDStateSpace, d::AbstractIdData, x0=zeros(res.sys.nx); stochastic=false)
	sys = res.sys
	@unpack A,B,C,D,ny = sys
	@unpack K,Q,R,P = res
	kf = KalmanFilter(res,x0)
	u = input(d)
	yh = map(observations(u,u)) do (ut,_)
		predict!(kf, ut)
		yh = kf.C*state(kf)
		stochastic ? StaticParticles(MvNormal(yh,Symmetric(C*covariance(kf)*C' + kf.R2))) : yh
	end
	oftype(u,yh)
end

m2vv(x) = [x[:,i] for i in 1:size(x,2)]
function predict(res::N4SIDStateSpace, d::AbstractIdData, x0=res.x[:,1])
	y = output(d)
	u = input(d)
	sys = res.sys
	@unpack C = sys
	kf = KalmanFilter(res,x0)
	X = forward_trajectory(kf, m2vv(u), m2vv(y))[1]
	size(X)
	yh = Ref(C).*X
	oftype(y,yh)
end

function ControlSystems.lsim(res::N4SIDStateSpace, u; x0=res.x[:,1])
	simulate(res.sys, input(u), x0)
end


function LowLevelParticleFilters.KalmanFilter(res::N4SIDStateSpace, x0=res.x[:,1])
	sys = res.sys
	@unpack A,B,C,D,ny = sys
	@unpack K,Q,R,P = res
	norm(D) < 0.1*norm(C) || throw(ArgumentError("Nonzero D matrix not supported yet"))
	kf = KalmanFilter(A, B, C, 0*D, Q, R, MvNormal(x0,P))
end

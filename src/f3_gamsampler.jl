
@with_kw struct GAMOptions
    P::Int64 = 2
    N::Int64 = 100
    γ::Vector{Float64} = [(0.99 / t) for t in 1:N]
    U::Vector{Float64} = ones(3) / 3
    PC::Int64 = 1
    λ0::Vector{Vector{Float64}} = [[1.0], ones(P), ones(PC)]
    μ0::Vector{Float64} = zeros(P)
    Σ0::Matrix{Float64} = diagm(0 => ones(P))
    eigval0::Float64 = 1.0
    eigvec0::Vector{Float64} = ones(P)
    PCstep::Int64 = 10
    ᾱ::Float64 = 0.20
    burn::Int64 = 0
end
# TODO: Write documentation for gamsampler function and related objects

struct GAMResults
    algorithm::String
    sample::Matrix{Float64}
    density::Vector{Float64}
    mode::Vector{Float64}
    N::Int64
    γ::Vector{Float64}
    λ::Vector{Matrix{Float64}}
    α::Vector{Matrix{Float64}}
    μ::Vector{Matrix{Float64}}
    Σ::Vector{Array{Float64,3}}
    eigval::Matrix{Float64}
    eigvec::Array{Float64,3}
end

function gamsampler(f::Function, x0::Vector{Float64}, opt::GAMOptions;
    saveproposal::Bool=false)

    @unpack N, γ, U, PC, λ0, μ0, Σ0, ᾱ, eigval0, eigvec0, burn, PCstep = opt

    # number of algorithms
    A = 3

    # assert sizes are consistent
    P = length(x0)
    @assert length(γ) == N "Vector γ must be N × 1."
    @assert size(Σ0) == (P, P) "Sizes of μ0 and Σ0 must be consistent."
    @assert PC <= P "Cannot compute more principal components than parameters."

    # prelims
    Gaussian = MvNormal(diagm(0 => ones(P)))
    p2::Int = 1 # stores parameter being updated by componentwise algorithm (a = 2)
    k3::Int = 1 # stores component being updated by pc algorithm (a = 3)
    cumU = cumsum(U)

    # pre-allocate output
    sample = zeros(N, P)
    density = zeros(N)

    # simulation
    x = x0
    d = f(x)
    λ_n = λ0
    α_n = ᾱ .* [[1], ones(P), ones(PC)]
    μ_n = [μ0 for _ in 1:A]
    sqΣ_n = [collect(cholesky(Σ0).L) for _ in 1:A]
    eigval_n = eigval0 * ones(PC)
    eigvec_n = eigvec0 * ones(PC)'

    # pre-allocate proposal
    if saveproposal
        α = [zeros(N, 1), zeros(N, P), zeros(N, PC)]
        λ = [zeros(N, 1), zeros(N, P), zeros(N, PC)]
        μ = [zeros(N, P) for _ in 1:A]
        Σ = [zeros(N, P, P) for _ in 1:A]
        eigval = zeros(N, PC)
        eigvec = zeros(N, P, PC)
        for a in 1:A
            α[a][1, :] = α_n[a]
            λ[a][1, :] = λ_n[a]
            μ[a][1, :] = μ_n[a]
            Σ[a][1, :, :] = sqΣ_n[a] * sqΣ_n[a]'
        end
        eigval[1, :] = eigval_n
        eigvec[1, :, :] = eigvec_n
    end

    display(md"**Generalized Adaptive Metropolis (AM) Sampler**")
    progress = Progress(N, dt=1.0, barlen=25)
    for n in 1:N

        # draw algorithm
        a = count(rand() .> cumU) + 1

        # select a candidate 
        if a == 1 # global adaptation

            # draw candidate and accept/reject
            jump = sqrt(λ_n[a][1]) * sqΣ_n[a] * rand(Gaussian)
            xc = x .+ jump

        elseif a == 2 # componentwise adaptation

            # draw parameter to update
            p2 = rand(1:P)

            # draw candidate
            sig = sqΣ_n[a] * sqΣ_n[a]'
            jump = sqrt(λ_n[a][p2]) * sqrt(sig[p2, p2]) * randn()
            xc = copy(x)
            xc[p2] = xc[p2] + jump

        elseif a == 3 # principal component adaptation

            # draw principal component
            k3 = rand(1:PC)

            # draw candidate
            jump = sqrt(λ_n[a][k3]) * sqrt(eigval_n[k3]) * randn()
            xc = x .+ eigvec_n[:, k3] * jump

        end

        # accept or reject proposal
        dc = f(xc)
        acceptrate = min(1, exp(dc - d))
        if rand() < acceptrate
            x = xc
            d = dc
        end
        sample[n, :] = x
        density[n] = d

        # adaptation: update proposal
        if n < N
            γ_n = γ[n+1]

            # parameter updated only if corresponding algorithm was selected:
            if a == 1
                λ_n[a][1] = λ_n[a][1] + γ_n * (acceptrate - ᾱ)
                α_n[a][1] = α_n[a][1] + γ_n * (acceptrate - α_n[a][1])
            elseif a == 2
                λ_n[a][p2] = λ_n[a][p2] + γ_n * (acceptrate - ᾱ)
                α_n[a][p2] = α_n[a][p2] + γ_n * (acceptrate - α_n[a][p2])
            elseif a == 3
                λ_n[a][k3] = λ_n[a][k3] + γ_n * (acceptrate - ᾱ)
                α_n[a][k3] = α_n[a][k3] + γ_n * (acceptrate - α_n[a][k3])
            end

            # parameters updated regardless of selected algorithm
            for a in 1:A
                mu = copy(μ_n[a])
                sqsig = copy(sqΣ_n[a])

                if U[a] > 0
                    updapprox!(mu, U[a] * γ_n, x)
                    updateΣ!(sqsig, mu, U[a] * γ_n, x)
                end
                μ_n[a] = mu
                sqΣ_n[a] = sqsig

                # save proposal
                if saveproposal
                    λ[a][n+1, :] = λ_n[a]
                    α[a][n+1, :] = α_n[a]
                    μ[a][n+1, :] = mu
                    Σ[a][n+1, :, :] = sqsig * sqsig'
                end
            end

            # update eigenvectors
            if (U[3] > 0.0) .& (rem(n, PCstep) == 0)
                sig = sqΣ_n[3] * sqΣ_n[3]'
                eig = eigen(sig, sortby=(x -> -x))
                eigval_n = eig.values[1:PC]
                eigvec_n = eig.vectors[:, 1:PC]
            end
            if saveproposal
                eigval[n+1, :] = eigval_n
                eigvec[n+1, :, :] = eigvec_n
            end
        end
        next!(progress,
            showvalues=[
                ("Log Density", d),
                ("Acceptance Rate (Global)", α_n[1][1]),
                [("Acceptance Rate (Pr Comp = $k)", α_n[3][k]) for k in 1:PC]...,
                [("Acceptance Rate (p = $p)", α_n[2][p]) for p in 1:P]...
            ])

    end
    println("")
    println("")

    # burn and build results
    algorithm = "Generalized Adaptive Metropolis"
    sample = sample[burn+1:end, :]
    density = density[burn+1:end]
    mode = sample[argmax(density), :]
    γ = γ[burn+1, :]
    if saveproposal
        for a in 1:A
            λ[a] = λ[a][burn+1:end, :]
            α[a] = α[a][burn+1:end, :]
            μ[a] = μ[a][burn+1:end, :]
            Σ[a] = Σ[a][burn+1:end, :, :]
        end
        eigval = eigval[burn+1:end, :]
        eigvec = eigvec[burn+1:end, :, :]
    else
        λ = [NaN * ones(1, 1) for _ in 1:A]
        α = [NaN * ones(1, 1) for _ in 1:A]
        μ = [NaN * ones(1, P) for _ in 1:A]
        Σ = [NaN * ones(1, P, P) for _ in 1:A]
        eigval = NaN * ones(1, PC)
        eigvec = NaN * ones(1, P, PC)
    end

    R = GAMResults(algorithm, sample, density, mode, N - burn, γ,
        λ, α, μ, Σ, eigval, eigvec)
    return R
end

summarytable(R::GAMResults) = summarytable(R.sample, R.density, R.Σ[1][end, :, :])
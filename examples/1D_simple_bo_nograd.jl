"""
This short example shows a 1D optimization of a function using the Bayesian Optimization framework.

f : R → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using JLD2
using BayesOpt
using LinearAlgebra
using LaTeXStrings
import Random
Random.seed!(555)

# Objective Function
f(x) = sin(sum(x.+1)) + sin((10.0 / 3.0) * sum(x .+1))
min_f = −1.988699758534924
problem_dim = 1
lower = [-10.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

σ² = 1e-6 # 1e-10

kernel = Matern52Kernel()
model = StandardGP(kernel, σ²) # Instantiates the StandardGP (gives it the prior).

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]

println(x_train)
y_train = f.(x_train) #+ sqrt(σ²).* randn(n_train);
y_train = map(x -> [x], y_train)
println(y_train)

using Optim
# Negative log marginal likelihood (no noise term)
function nlml(params,kernel,X_train,y_train,σ²)
    log_ℓ, log_scale = params
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    # Kernel with current parameters
    k = scale * (kernel ∘ ScaleTransform(ℓ))
    mod = StandardGP(k, σ²)

    #println(mean(gp.gpx(x_train)))

    -AbstractGPs.logpdf(mod.gp(X_train,σ²), reduce(vcat,y_train))  # Negative log marginal likelihood
end

# Initial log-parameters: log(lengthscale), log(magnitude)
initial_params = [log(1.0), log(1.0)]

# Optimize with BFGS
res = Optim.optimize(x -> nlml(x,kernel,x_train,y_train,σ²), initial_params, Optim.Newton())

# Extract optimized values
opt_params = Optim.minimizer(res)
ell_opt = exp(opt_params[1])
scale_opt = exp(opt_params[2])

println("Optimized lengthscale: ", ell_opt)
println("Optimized magnitude: ", scale_opt)

kernel = scale_opt *(kernel ∘ ScaleTransform(ell_opt))
model = StandardGP(kernel, σ²)

# Conditioning: 
# We are conditionning the GP, returning GP|X,y where y can be noisy (but supposed fixed anyway)
model = update!(model, x_train, y_train)


# Init of the acquisition function
ξ = 1e-3
acqf = ExpectedImprovement(ξ, minimum(reduce(vcat,y_train)))
#acqf = KnowledgeGradient(domain, [optimize_mean!(model, domain)[2]])

#plot_domain = collect(lower[1]:0.1:upper[1])
#acqf_dom = [acqf(model,x) for x in plot_domain]
#plot(plot_domain,acqf_dom)

#plot(plot_domain,f.(plot_domain))
#line!(optimize_mean!(model, domain))

# This maximises the function
problem = BOProblem(
                    f,
                    domain,
                    model,
                    copy(x_train),
                    copy(y_train),
                    acqf,
                    90,
                    0.0
                    )

print_info(problem)

@info "Starting Bayesian Optimization..."
result = BayesOpt.optimize(problem)
xs = reduce(vcat,result.xs)
ys = reduce(vcat,result.ys)

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))


K̃ = kernelmatrix(kernel,xs)+σ²*I(length(xs))
κ_K = cond(K̃)

@load "grad_bo_1d.jld2" feval_grad error_grad

running_min = accumulate(min, f.(xs))

Plots.plot(1:length(running_min),err,yaxis=:log, title="Error w.r.t true minimum (1D BO)",
            xlabel="Function evaluations",ylabel=L"|| f(x^*_n) - f^* ||",
            label="BO",xlims=(1,length(running_min)))
Plots.vspan!([1,n_train]; color=:blue,alpha=0.2, label="")
Plots.vspan!([n_train,2*n_train]; color=:purple,alpha=0.2, label="")
Plots.plot!(feval_grad,error_grad,label="gradBO",yaxis=:log)
savefig("1D_error_BO.pdf")

plot_domain = collect(lower[1]:0.01:upper[1])

plot_domain = prep_input(model,plot_domain)

post_mean, post_var = mean_and_var(result.gp.gpx(plot_domain))

plot(plot_domain, f.(plot_domain),
        label="target function",
        xlim=(lower[1], upper[1]),
        xlabel="x",
        ylabel="y",
        title="BayesOpt, EI ξ=$(ξ), σ²=$(σ²)",
        legend=:outertopright)
plot!(plot_domain, post_mean; label="GP", ribbon=sqrt.(abs.(post_var)),ribbon_scale=2,color="green")
scatter!(
    xs[1:n_train],
    ys[1:n_train];
    label="Train Data"
)
scatter!(
    xs[n_train+1:end],
    ys[n_train+1:end];
    label="Candidates"
)
scatter!(
    [xs[argmin(ys)]],
    [minimum(ys)];
    label="Best candidate"
)
savefig("gp_Matern_1D.pdf")
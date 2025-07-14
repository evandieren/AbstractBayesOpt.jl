"""
This short example shows a 1D optimization of a function using the Bayesian Optimization framework.

f : R → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using BayesOpt
using Optim
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

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]

y_train = f.(x_train) #+ sqrt(σ²).* randn(n_train);
y_train = map(x -> [x], y_train)

kernel_constructor = Matern52Kernel()

# # Initial log-parameters: log(lengthscale), log(magnitude)
# initial_params = [log(1.0), log(1.0)]

# # Optimize with BFGS
# res = Optim.optimize(x -> nlml(x,kernel,x_train,y_train,σ²), initial_params, Optim.Newton())

# # Extract optimized values
# opt_params = Optim.minimizer(res)
# ell_opt = exp(opt_params[1])
# scale_opt = exp(opt_params[2])

# println("Optimized lengthscale: ", ell_opt)
# println("Optimized magnitude: ", scale_opt)

kernel = 1 *(kernel_constructor ∘ ScaleTransform(1))
model = StandardGP(kernel, σ²,mean=ConstMean(mean(reduce(vcat, y_train))))

# Conditioning: 
model = update!(model, x_train, y_train)


# Init of the acquisition function
ξ = 1e-3
acqf = ExpectedImprovement(ξ, minimum(reduce(vcat,y_train)))

# This maximises the function
problem = BOProblem(
                    f,
                    domain,
                    model,
                    kernel_constructor,
                    copy(x_train),
                    copy(y_train),
                    acqf,
                    90,
                    0.0
                    )

print_info(problem)

@info "Starting Bayesian Optimization..."
result, acq_list = BayesOpt.optimize(problem)
xs = reduce(vcat,result.xs)
ys = reduce(vcat,result.ys)

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))


running_min = accumulate(min, f.(xs))

p = Plots.plot(n_train:length(running_min),running_min[n_train:end] .- min_f,yaxis=:log, title="Error w.r.t true minimum (1D BO)",
            xlabel="Function evaluations",ylabel=L"|| f(x^*_n) - f^* ||",
            label="BO",xlims=(1,length(running_min)))
Plots.vspan!([1,n_train]; color=:blue,alpha=0.2, label="")
Plots.display(p)
# savefig("examples/1D_error_BO.pdf")

plot_domain = collect(lower[1]:0.01:upper[1])
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
# savefig("examples/gp_Matern_1D.pdf")
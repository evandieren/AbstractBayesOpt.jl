"""
This short example shows a 1D optimization of a function using the Bayesian Optimization framework.

f : R → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using ForwardDiff
using BayesOpt
using LinearAlgebra
import Random
using Optim
Random.seed!(555)

# Objective Function
f(x) = sin(sum(x.+1)) + sin((10.0 / 3.0) * sum(x .+1))
∇f(x) = ForwardDiff.gradient(f, x)
min_f = −1.988699758534924
f_val_grad(x) = [f(x); ∇f(x)]

d = 1
lower = [-10.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

grad_kernel = gradKernel(ApproxMatern52Kernel())

σ² = 1e-6
model = GradientGP(grad_kernel,d+1,σ²)

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(d) for _ in 1:n_train]
println(x_train)
val_grad = f_val_grad.(x_train)
# Create flattened output
y_train = [val_grad[i] for i = eachindex(val_grad)]
#y_train = [val_grad[i] + sqrt(σ²)*randn(d+1) for i = eachindex(val_grad)]

println(y_train)

# Initial log-parameters: log(lengthscale), log(magnitude)
initial_params = [log(1.0), log(1.0)]

# Optimize with BFGS
res = Optim.optimize(x -> nlml_grad(x,ApproxMatern52Kernel(),x_train,y_train,σ²), initial_params, Optim.Newton())

# Extract optimized values
opt_params = Optim.minimizer(res)
ell_opt = exp(opt_params[1])
scale_opt = exp(opt_params[2])

println("Optimized lengthscale: ", ell_opt)
println("Optimized magnitude: ", scale_opt)

grad_kernel = gradKernel(scale_opt *(ApproxMatern52Kernel() ∘ ScaleTransform(ell_opt)))
model = GradientGP(grad_kernel,d+1,σ²)
# Conditioning: 
# We are conditionning the GP, returning GP|X,y where y can be noisy (but supposed fixed anyway)
model = update!(model, x_train, y_train)

# Init of the acquisition function
ξ = 1e-3
acqf = ExpectedImprovement(ξ, minimum(hcat(y_train...)[1,:]))

# This maximises the function
problem = BOProblem(
                    f_val_grad,
                    domain,
                    model,
                    copy(x_train),
                    copy(y_train),
                    acqf,
                    40,
                    0.0
                    )

print_info(problem)

@info "Starting Bayesian Optimization..."
result = BayesOpt.optimize(problem)
xs = reduce(vcat,result.xs)
ys = hcat(result.ys...)[1,:]
println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))

running_min = accumulate(min, f.(xs))

running_min = collect(Iterators.flatten(fill(x, 2) for x in (running_min)))


p = Plots.plot((2*n_train):length(running_min),running_min[2*n_train:end] .- min_f,yaxis=:log, title="Error w.r.t true minimum (1D GradBO)",
            xlabel="Function evaluations",ylabel=L"|| f(x^*_n) - f^* ||",
            label="GradBO",xlims=(1,length(running_min)))
Plots.vspan!([1,2*n_train]; color=:blue,alpha=0.2, label="")
Plots.display(p)


plot_domain = collect(lower[1]:0.01:upper[1])

plot_x = map(x -> [x], plot_domain)
plot_x = prep_input(model,plot_x)
post_mean, post_var = mean_and_var(result.gp.gpx(plot_x))

post_mean = reshape(post_mean, :, d+1)[:,1] # This returns f(x) to match the StandardGP
post_var = reshape(post_var, :, d+1)[:,1]
post_var[post_var .< 0] .= 0

plot(plot_domain, f.(plot_domain),
        label="target function",
        xlim=(lower[1], upper[1]),
        xlabel="x",
        ylabel="y",
        title="BayesOpt, σ²=$(σ²)",
        legend=:outertopright)
plot!(plot_domain, post_mean; label="GP", ribbon=sqrt.(post_var),ribbon_scale=2,color="green")
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
#savefig("gradgp_matern_1D_ensemble_EI_ucb_grad.pdf")
"""
This short example shows a 1D optimization of a function using the Bayesian Optimization framework.

f : R → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using ForwardDiff
using LinearAlgebra
using LaTeXStrings
import Random
using Optim
using AbstractBayesOpt
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


σ² = 1e-12

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(d) for _ in 1:n_train]
println(x_train)
val_grad = f_val_grad.(x_train)
# Create flattened output
y_train = [val_grad[i] for i = eachindex(val_grad)]

kernel_constructor = ApproxMatern52Kernel()

kernel = 1 *(kernel_constructor ∘ ScaleTransform(1))
grad_kernel = gradKernel(kernel)
model = GradientGP(grad_kernel,d+1,σ²)

# Init of the acquisition function
ξ = 0.0
acqf = ExpectedImprovement(ξ, minimum(hcat(y_train...)[1,:]))

# This maximises the function
bo_struct = BOStruct(
                    f_val_grad,
                    acqf,
                    model,
                    domain,
                    copy(x_train),
                    copy(y_train),
                    100,
                    0.0
                    )

print_info(bo_struct)

@info "Starting Bayesian Optimization..."
choice = "mean_only" 
result, acq_list, standard_params = AbstractBayesOpt.optimize(bo_struct, standardize=choice)
xs = reduce(vcat,result.xs)
ys = result.ys_non_std 
ys = hcat(ys...)[1,:]
println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))

# xs_nothing = copy(xs)
# acq_nothing = copy(acq_list)

# xs_scale_only = copy(xs)
# acq_scale_only = copy(acq_list .* standard_params[2][1])

# xs_mean_scale = copy(xs)
# acq_mean_scale = copy(acq_list .* standard_params[2][1])

# xs_mean_only = copy(xs)
# acq_mean_only = copy(acq_list)

# plot(max.(acqf_list,1e-13),yaxis=:log)
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
post_mean, post_var = unstandardized_mean_and_var(result.model,plot_x, standard_params)

post_mean = reshape(post_mean, :, d+1)[:,1] # This returns f(x) to match the StandardGP
post_var = reshape(post_var, :, d+1)[:,1]
post_var[post_var .< 0] .= 0

plot(plot_domain, f.(plot_domain),
        label="target function",
        xlim=(lower[1], upper[1]),
        xlabel="x",
        ylabel="y",
        title="AbstractBayesOpt, σ²=$(σ²)",
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

# Plots.plot(n_train:length(acq_nothing), acq_nothing[n_train:end] .+ eps(), label="standardize = nothing", xlabel="Iteration",
#         ylabel="Acquisition value", title="Acquisition value over iterations (1D BO)", yaxis=:log)
# Plots.plot!(n_train:length(acq_scale_only), acq_scale_only[n_train:end] .+ eps(), label="standardize = scale_only",ls=:dash)

# Plots.plot(n_train:length(acq_mean_only), acq_mean_only[n_train:end] .+ eps(), label="standardize = mean_only",ls=:dot,yaxis=:log)
# Plots.plot!(n_train:length(acq_mean_scale), acq_mean_scale[n_train:end] .+ eps(), label="standardize = mean_scale",ls=:dashdot)

# Plots.plot(n_train:length(xs_nothing), f.(xs_nothing)[n_train:end], label="standardize = nothing", xlabel="Iteration", ylabel="f(x)", title="Value of f at sampled points (1D BO)")
# Plots.plot!(n_train:length(xs_scale_only), f.(xs_scale_only)[n_train:end], label="standardize = scale_only",ls=:dash)

# Plots.plot(n_train:length(xs_mean_only), f.(xs_mean_only)[n_train:end], label="standardize = mean_only",ls=:dot)
# Plots.plot!(n_train:length(xs_mean_scale), f.(xs_mean_scale)[n_train:end], label="standardize = mean_scale",ls=:dashdot)


# Identical without hyper parameter tuning 
# When I add hyper parameter tuning, the results are barely identical, one or two points are not the same.
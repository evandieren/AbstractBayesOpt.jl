"""
This short example shows a Rᵈ optimization of a function using the Bayesian Optimization framework.

f : Rᵈ → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using ForwardDiff
#using GLMakie
using LinearAlgebra

using BayesOpt

import Random
Random.seed!(1234)

# Objective Function 
function ST(x::AbstractVector) # Styblinski–Tang function
    0.5*sum(x.^4 - 16 .*x.^2 + 5 .*x )
end

f(x) = ST(x)

∇f(x) = ForwardDiff.gradient(f, x)

f_val_grad(x) = [f(x); ∇f(x)]

problem_dim = 10
lower = -5 .*ones(problem_dim)
upper = 5 .*ones(problem_dim)
domain = ContinuousDomain(lower, upper)

grad_kernel = gradKernel(ApproxMatern52Kernel())
model = GradientGP(grad_kernel,problem_dim+1)

# Generate uniform random samples
n_train = 2^10
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]
println(x_train)

σ² = 0.0
val_grad = f_val_grad.(x_train)
# Create flattened output
y_train = [val_grad[i] + sqrt(σ²)*randn(problem_dim+1) for i = eachindex(val_grad)]

println(y_train)

# Conditioning: 
# We are conditionning the GP, returning GP|X,y where y can be noisy (but supposed fixed)
model = update!(model, x_train, y_train, σ²)

# Init of the acquisition function
ξ = 1e-3
acqf = ExpectedImprovement(ξ, minimum(hcat(y_train...)[1,:]))

# This maximises the function
problem = BOProblem(
                    f_val_grad, # because we probe both the function value and its gradients.
                    domain,
                    model,
                    copy(x_train),
                    copy(y_train),
                    acqf,
                    30,
                    σ²
                    )

print_info(problem)

@info "Starting Bayesian Optimization..."
result = optimize(problem)
xs = result.xs #reduce(vcat,result.xs)
ys = hcat(result.ys...)[1,:]

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))

running_min = accumulate(min, f.(xs))[n_train+1:end]
Plots.plot(1:length(running_min), norm.(running_min .- f_star),yaxis=:log, title="Error w.r.t true minimum ($(problem_dim)-D BO)")
savefig("$(problem_dim)_D_error_bo_OUU_grad.pdf")

# @info "Starting plotting procedure..."
# plot_grid_size = 250 # Grid for surface plot.
# x_grid = range(domain.lower[1], domain.upper[1], length=plot_grid_size)
# y_grid = range(domain.lower[2], domain.upper[2], length=plot_grid_size)

# grid_values = [f([x,y]) for x in x_grid, y in y_grid]
# grid_mean = [mean(result.gp.gpx([([x,y],1)]))[1] for x in x_grid, y in y_grid]
# grid_std = sqrt.([var(result.gp.gpx([([x,y],1)]))[1] for x in x_grid, y in y_grid])
# grid_acqf = [result.acqf(result.gp,[x,y]) for x in x_grid, y in y_grid]

# fig = Figure(;size=(1000, 600))
# ax1 = Axis(fig[1, 1], title="True function", xlabel="X-axis", ylabel="Y-axis")
# ax2 = Axis(fig[1, 2], title="Posterior mean", xlabel="X-axis", ylabel="Y-axis")
# ax3 = Axis(fig[2, 1], title="Posterior standard deviation", xlabel="X-axis", ylabel="Y-axis")
# ax4 = Axis(fig[2, 2], title="Acquisition function (EI)", xlabel="X-axis", ylabel="Y-axis")

# #GLMakie.contour!(ax1, x_grid, y_grid, grid_values, colormap=:viridis, levels=200)
# x1_coords = hcat(xs...)[1,:]
# x2_coords = hcat(xs...)[2,:]

# # True function evaluation
# GLMakie.surface!(ax1,x_grid, y_grid,fill(0f0, size(grid_values));
#                  color=grid_values, shading = NoShading)
# GLMakie.scatter!(ax1, x1_coords, x2_coords)
# GLMakie.scatter!(ax1, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],color="green")
# Colorbar(fig[1, 1][1, 2],#scale=log10,
#          limits = (minimum(grid_values),maximum(grid_values)))

# # Posterior mean
# GLMakie.surface!(ax2,x_grid, y_grid,fill(0f0, size(grid_mean));
#                  color=grid_mean, shading = NoShading,
#                  colorrange = (minimum(grid_values),maximum(grid_values)))
# GLMakie.scatter!(ax2, x1_coords, x2_coords)
# GLMakie.scatter!(ax2, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],color="green")
# Colorbar(fig[1, 2][1, 2],#scale=log10,
#          limits = (minimum(grid_values),maximum(grid_values)))

# # Posterior variance
# GLMakie.surface!(ax3,x_grid, y_grid,fill(0f0, size(grid_std));
#                  color=grid_std, shading = NoShading)
# GLMakie.scatter!(ax3, x1_coords, x2_coords)
# GLMakie.scatter!(ax3, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],color="green")
# Colorbar(fig[2, 1][1, 2],limits = (minimum(grid_std),maximum(grid_std)))

# # Acquisition function
# GLMakie.surface!(ax4,x_grid, y_grid,fill(0f0, size(grid_acqf)); 
#                  color=grid_acqf, shading = NoShading)
# GLMakie.scatter!(ax4, x1_coords, x2_coords)
# GLMakie.scatter!(ax4, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],color="green")
# Colorbar(fig[2, 2][1, 2],limits = (minimum(grid_acqf),maximum(grid_acqf)))

# #savefig(fig,"output_example_2D.png")
# GLMakie.activate!(inline=true)
# display(fig)
# save("gradgp_matern_ST_2D.png",fig)
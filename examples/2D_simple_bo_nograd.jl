"""
This short example shows a 2D optimization of a function using the Bayesian Optimization framework.

f : R² → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using GLMakie

using BayesOpt

import Random
Random.seed!(1234)

# Objective Function 
# Branin
function branin(x::AbstractVector)
    x1 = x[1]
    x2 = x[2]
    b = 5.1 / (4*pi^2);
    c = 5/pi;
    r = 6;
    a = 1;
    s = 10;
    t = 1 / (8*pi);
    term1 = a * (x2 - b*x1^2 + c*x1 - r)^2;
    term2 = s*(1-t)*cos(x1);
    y = term1 + term2 + s;
end

# Rosenbrock
rosenbrock(x::AbstractVector) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

# Himmelblau
himmelblau(x::AbstractVector) = (x[1]^2 + x[2] -11)^2 + (x[1]+x[2]^2-7)^2

f(x) = himmelblau(x)

problem_dim = 2
lower = [-6,-6.0] #[-5.0, 0.0]
upper = [6.0,6.0] #[10.0, 15.0]
domain = ContinuousDomain(lower, upper)

kernel = Matern52Kernel()
model = StandardGP(kernel) # Instantiates the StandardGP (gives it the prior).

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]
σ² = 0.0
y_train = f.(x_train) + sqrt(σ²).* randn(n_train)
y_train = map(x -> [x], y_train)

println(y_train)

#x_train = hcat(x_train...) # easier for AbstractGPs to work with Abstract Matrices

println(x_train)


# Conditioning: 
# We are conditionning the GP, returning GP|X,y where y can be noisy (but supposed fixed)
model = update!(model, x_train, y_train, σ²)

# Init of the acquisition function
ξ = 1e-3
acqf = ExpectedImprovement(ξ, minimum(y_train)[1])

# This maximises the function
problem = BOProblem(
                    f,
                    domain,
                    model,
                    x_train,
                    y_train,
                    acqf,
                    30,
                    σ²
                    )

print_info(problem)

@info "Starting Bayesian Optimization..."
result = optimize(problem)
xs = result.xs
ys = result.ys

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))

@info "Starting plotting procedure..."
plot_grid_size = 250 # Grid for surface plot.
x_grid = range(domain.lower[1], domain.upper[1], length=plot_grid_size)
y_grid = range(domain.lower[2], domain.upper[2], length=plot_grid_size)

grid_values = [f([x,y]) for x in x_grid, y in y_grid]
grid_mean = [mean(result.gp.gpx([[x,y]]))[1] for x in x_grid, y in y_grid]
grid_std = sqrt.([var(result.gp.gpx([[x,y]]))[1] for x in x_grid, y in y_grid])
grid_acqf = [result.acqf(result.gp,[x,y]) for x in x_grid, y in y_grid]

fig = Figure(;size=(1000, 600))
ax1 = Axis(fig[1, 1], title="True function", xlabel="X-axis", ylabel="Y-axis")
ax2 = Axis(fig[1, 2], title="Posterior mean", xlabel="X-axis", ylabel="Y-axis")
ax3 = Axis(fig[2, 1], title="Posterior standard deviation", xlabel="X-axis", ylabel="Y-axis")
ax4 = Axis(fig[2, 2], title="Acquisition function (EI)", xlabel="X-axis", ylabel="Y-axis")

#GLMakie.contour!(ax1, x_grid, y_grid, grid_values, colormap=:viridis, levels=200)
x1_coords = hcat(xs...)[1,:]
x2_coords = hcat(xs...)[2,:]

# True function evaluation
GLMakie.surface!(ax1,x_grid, y_grid,fill(0f0, size(grid_values));
                 color=grid_values, shading = NoShading)
GLMakie.scatter!(ax1, x1_coords, x2_coords)
GLMakie.scatter!(ax1, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],color="green")
Colorbar(fig[1, 1][1, 2],scale=log10,
         limits = (minimum(grid_values),maximum(grid_values)))

# Posterior mean
GLMakie.surface!(ax2,x_grid, y_grid,fill(0f0, size(grid_mean));
                 color=grid_mean, shading = NoShading,
                 colorrange = (minimum(grid_values),maximum(grid_values)))
GLMakie.scatter!(ax2, x1_coords, x2_coords)
GLMakie.scatter!(ax2, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],color="green")
Colorbar(fig[1, 2][1, 2],scale=log10,
         limits = (minimum(grid_values),maximum(grid_values)))

# Posterior variance
GLMakie.surface!(ax3,x_grid, y_grid,fill(0f0, size(grid_std));
                 color=grid_std, shading = NoShading)
GLMakie.scatter!(ax3, x1_coords, x2_coords)
GLMakie.scatter!(ax3, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],color="green")
Colorbar(fig[2, 1][1, 2],limits = (minimum(grid_std),maximum(grid_std)))

# Acquisition function
GLMakie.surface!(ax4,x_grid, y_grid,fill(0f0, size(grid_acqf)); 
                 color=grid_acqf, shading = NoShading)
GLMakie.scatter!(ax4, x1_coords, x2_coords)
GLMakie.scatter!(ax4, [xs[argmin(ys)][1]], [xs[argmin(ys)][2]],color="green")
Colorbar(fig[2, 2][1, 2],limits = (minimum(grid_acqf),maximum(grid_acqf)))

#savefig(fig,"output_example_2D.png")
GLMakie.activate!(inline=true)
display(fig)
save("gp_matern_2D.png",fig)
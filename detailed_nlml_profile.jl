"""
Detailed profiling of NLML computation to identify specific bottlenecks.
Compares original GradientGP vs OptimizedGradientGP performance.
"""

using AbstractGPs
using KernelFunctions
using ForwardDiff
using BayesOpt
using BenchmarkTools
using LinearAlgebra
using Random

Random.seed!(42)

# Test function and its gradient
f(x) = sin(sum(x .+ 1)) + sin((10.0 / 3.0) * sum(x .+ 1))
∇f(x) = ForwardDiff.gradient(f, x)
f_val_grad(x) = [f(x); ∇f(x)]

# Problem setup
d = 2  # 2D problem
lower = [-10.0, -10.0]
upper = [10.0, 10.0]
σ² = 1e-12
n_train = 15
x_train = [lower .+ (upper .- lower) .* rand(d) for _ in 1:n_train]

# Standard GP data
y_train_standard = f.(x_train)
y_train_standard = map(x -> [x], y_train_standard)

# Gradient GP data
y_train_gradient = f_val_grad.(x_train)

# Kernel constructor
kernel_constructor = ApproxMatern52Kernel()
old_params = [log(1.0), log(1.0)]

# Setup GPs
kernel_standard = 1 * (kernel_constructor ∘ ScaleTransform(1))
gp_standard = StandardGP(kernel_standard, σ²)
gp_standard = update!(gp_standard, x_train, y_train_standard)

kernel_gradient = 1 * (kernel_constructor ∘ ScaleTransform(1))
grad_kernel = gradKernel(kernel_gradient)
gp_gradient = GradientGP(grad_kernel, d+1, σ²)
gp_gradient = update!(gp_gradient, x_train, y_train_gradient)

# ============================================================================
# Break down the NLML computation step by step
# ============================================================================

println("\n1. Data preparation comparison:")

function profile_data_prep_standard()
    x_prepped = prep_input(gp_standard, x_train)
    y_prepped = reduce(vcat, y_train_standard)
    return x_prepped, y_prepped
end

function profile_data_prep_gradient()
    x_prepped = prep_input(gp_gradient, x_train)
    y_prepped = vec(permutedims(reduce(hcat, y_train_gradient)))
    return x_prepped, y_prepped
end

function profile_data_prep_optimized()
    x_prepped = prep_input(gp_optimized, x_train)
    y_prepped = vec(permutedims(reduce(hcat, y_train_gradient)))
    return x_prepped, y_prepped
end

println("Standard GP data prep:")
@time x_std, y_std = profile_data_prep_standard()
println("Data size: x=$(length(x_std)), y=$(length(y_std))")

println("Gradient GP data prep:")
@time x_grad, y_grad = profile_data_prep_gradient()
println("Data size: x=$(x_grad.x |> length), y=$(length(y_grad))")


println("\n2. GP construction comparison:")

function profile_gp_construction_standard()
    ℓ, scale = exp.(old_params)
    k = scale * (kernel_constructor ∘ ScaleTransform(1/ℓ))
    return StandardGP(k, σ²)
end

function profile_gp_construction_gradient()
    ℓ, scale = exp.(old_params)
    k = scale * (kernel_constructor ∘ ScaleTransform(1/ℓ))
    return GradientGP(gradKernel(k), d+1, σ²)
end

println("Standard GP construction:")
@time gp_test_std = profile_gp_construction_standard()

println("Gradient GP construction:")
@time gp_test_grad = profile_gp_construction_gradient()

println("\n3. FiniteGP creation comparison:")
function profile_finite_gp_standard()
    gp_test = profile_gp_construction_standard()
    return gp_test.gp(x_std, σ²)
end

function profile_finite_gp_gradient()
    gp_test = profile_gp_construction_gradient()
    return gp_test.gp(x_grad, σ²)
end
println("Standard GP FiniteGP creation:")
@time gpx_std = profile_finite_gp_standard()

println("Gradient GP FiniteGP creation:")
@time gpx_grad = profile_finite_gp_gradient()


println("\n4. LogPDF computation comparison:")

function profile_logpdf_standard()
    gpx = profile_finite_gp_standard()
    return AbstractGPs.logpdf(gpx, y_std)
end

function profile_logpdf_gradient()
    gpx = profile_finite_gp_gradient()
    return AbstractGPs.logpdf(gpx, y_grad)
end

function test(gp,Y) 
    m, C_mat = mean_and_cov(gp.gp(gp.gpx.data.x, gp.noise_var))
    #K_grad = kernelmatrix(gp_test_grad.gp.kernel, x_grad) + σ² * I
    C2 = cholesky(AbstractGPs._symmetric(C_mat))
    T = promote_type(eltype(m), eltype(C), eltype(Y))
    return -((size(Y, 1) * T(AbstractGPs.log2π) + logdet(C2)) .+ AbstractGPs._sqmahal(m, C2, Y)) ./ 2
end


@benchmark nlml(gp_gradient, [0.0,0.0], kernel_constructor, x_grad, y_grad)

nlml(gp_gradient, [0.0,0.0], kernel_constructor, x_grad, y_grad)


function fastnlml(gp,Y)
    m = mean(gp.gp(gp.gpx.data.x,gp.noise_var))
    T = promote_type(eltype(m), eltype(gp.gpx.data.C),eltype(Y))
    return -((size(Y, 1) * T(AbstractGPs.log2π) + logdet(gp.gpx.data.C)) .+ AbstractGPs._sqmahal(m, gp.gpx.data.C, Y)) ./ 2
end

@benchmark test2(gp_gradient, y_grad)

function compute_analytical_logpdf_components()
    
    # Get the current kernel matrix for gradient GP
    K_grad = kernelmatrix(gp_gradient.gp.kernel, x_grad) + σ² * I
    
    # Compute the three components of the logpdf
    K_inv_y = K_grad \ y_grad
    quadratic_form = y_grad' * K_inv_y
    log_det_term = logdet(K_grad)
    constant_term = length(y_grad) * log(2π)
    
    analytical_logpdf = -0.5 * (quadratic_form + log_det_term + constant_term)
    
    return analytical_logpdf, K_grad
end

@benchmark analytical_result, K_grad = compute_analytical_logpdf_components()

println("Standard GP logpdf:")
@benchmark logpdf_std = profile_logpdf_standard()

println("Gradient GP logpdf:")
@benchmark logpdf_grad = profile_logpdf_gradient()

# Above is the important benchmark



println("\n5. Matrix size analysis:")
println("Standard GP:")
println("  Kernel matrix size: $(length(x_std)) x $(length(x_std))")

# For gradient GP, extract the effective matrix size
total_grad_size = length(y_grad)
println("Gradient GP:")
println("  Kernel matrix size: $total_grad_size x $total_grad_size")
println("  Size ratio: $(total_grad_size^2 / length(x_std)^2)")

println("\n6. Kernel evaluation profiling:")

# Profile individual kernel evaluations
function profile_standard_kernel_eval()
    k = gp_test_std.gp.kernel
    count = 0
    for i in 1:length(x_std)
        for j in 1:length(x_std)
            k(x_std[i], x_std[j])
            count += 1
        end
    end
    return count
end

function profile_gradient_kernel_eval()
    k = gp_test_grad.gp.kernel
    # Gradient kernel works on (x, output_index) pairs
    count = 0
    for i in 1:length(x_train)
        for oi in 1:(d+1)
            for j in 1:length(x_train)
                for oj in 1:(d+1)
                    k((x_train[i], oi), (x_train[j], oj))
                    count += 1
                end
            end
        end
    end
    return count
end

println("Standard kernel evaluations (full matrix):")
@time std_evals = profile_standard_kernel_eval()
println("Number of evaluations: $std_evals")

println("Gradient kernel evaluations (full matrix):")
@time grad_evals = profile_gradient_kernel_eval()
println("Number of evaluations: $grad_evals")

println("\n7. Memory allocation during NLML:")
println("Standard GP NLML allocations:")
@time nlml(gp_standard, old_params, kernel_constructor, x_std, y_std)

println("Gradient GP NLML allocations:")
@time nlml(gp_gradient, old_params, kernel_constructor, x_grad, y_grad)


# Let's verify the kernel evaluation cost
println("\n8. Individual kernel evaluation cost:")

k_std = gp_test_std.gp.kernel
k_grad = gp_test_grad.gp.kernel


x1, x2 = x_train[1], x_train[2]

function std_kernel_single()
    for i in 1:1000
        k_std(x1, x2)
    end
end

function grad_kernel_single()
    for i in 1:1000
        k_grad((x1, 1), (x2, 1))  # function-function evaluation
    end
end

function grad_kernel_mixed()
    for i in 1:1000
        k_grad((x1, 1), (x2, 2))  # function-gradient evaluation
    end
end

function grad_kernel_grad_grad()
    for i in 1:1000
        k_grad((x1, 2), (x2, 2))  # gradient-gradient evaluation
    end
end


println("Standard kernel (1000 evals):")
@time std_kernel_single()

println("Gradient kernel f-f (1000 evals):")
@time grad_kernel_single()

println("Gradient kernel f-∇ (1000 evals):")
@time grad_kernel_mixed()

println("Gradient kernel ∇-∇ (1000 evals):")
@time grad_kernel_grad_grad()

# ...existing code...

println("\n9. Allocation source analysis:")

# Test direct kernel vs wrapped kernel
println("Direct kernel evaluation (no wrapper):")
k_direct = gp_test_std.gp.kernel
@allocated k_direct(x1, x2)

println("Gradient kernel wrapper overhead:")
# The gradient kernel adds tuple processing
input1 = (x1, 1)
input2 = (x2, 1)
@allocated k_grad(input1, input2)

# Let's also check if we can access the underlying kernel
println("\nGradient kernel internals:")
println("Gradient kernel fields: $(fieldnames(typeof(k_grad)))")

# Use BenchmarkTools for more accurate measurements
println("\n10. Proper benchmarking with warm-up:")

# Warm-up calls to avoid compilation overhead
println("Warming up functions...")
k_std(x1, x2)  # Warm-up
k_grad((x1, 1), (x2, 1))  # Warm-up

println("Standard kernel (post warm-up):")
@time begin
    for i in 1:1000
        k_std(x1, x2)
    end
end

println("Gradient kernel f-f (post warm-up):")
@time begin
    for i in 1:1000
        k_grad((x1, 1), (x2, 1))
    end
end

# Now try the direct access
try_direct_kernel_access()

println("\n12. Using @benchmark for accurate timing:")
# If you have BenchmarkTools loaded, use this for more accurate results
println("Standard kernel benchmark:")
@benchmark k_std($x1, $x2)

println("Gradient kernel f-f benchmark:")
@benchmark k_grad(($x1, 1), ($x2, 1))
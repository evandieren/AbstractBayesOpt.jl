using Test
using BayesOpt

@testset "Domain Tests" begin
    @testset "ContinuousDomain Construction" begin
        # Test valid domain construction
        lower = [0.0, -1.0]
        upper = [1.0, 1.0]
        domain = ContinuousDomain(lower, upper)
        
        @test domain.lower == lower
        @test domain.upper == upper
        @test domain.bounds == [(0.0, 1.0), (-1.0, 1.0)]
        
        # Test single dimension
        domain_1d = ContinuousDomain([0.0], [1.0])
        @test domain_1d.lower == [0.0]
        @test domain_1d.upper == [1.0]
        @test domain_1d.bounds == [(0.0, 1.0)]
    end
    
    @testset "ContinuousDomain Error Handling" begin
        # Test bounds mismatch
        @test_throws ArgumentError ContinuousDomain([0.0, 1.0], [1.0])
        @test_throws ArgumentError ContinuousDomain([0.0], [1.0, 2.0])
        
        # Test invalid bounds (lower > upper)
        @test_throws ArgumentError ContinuousDomain([1.0], [0.0])
        @test_throws ArgumentError ContinuousDomain([0.0, 2.0], [1.0, 1.0])
        
        # Test equal bounds (should be valid)
        domain_equal = ContinuousDomain([1.0], [1.0])
        @test domain_equal.lower == [1.0]
        @test domain_equal.upper == [1.0]
    end
    
    @testset "ContinuousDomain Edge Cases" begin
        # Test large values
        large_domain = ContinuousDomain([1e6], [1e7])
        @test length(large_domain.bounds) == 1
        
        # Test negative bounds
        neg_domain = ContinuousDomain([-10.0, -5.0], [-1.0, 0.0])
        @test neg_domain.bounds == [(-10.0, -1.0), (-5.0, 0.0)]
        
        # Test high dimensional domain
        n_dims = 10
        lower_nd = zeros(n_dims)
        upper_nd = ones(n_dims)
        domain_nd = ContinuousDomain(lower_nd, upper_nd)
        @test length(domain_nd.bounds) == n_dims
        @test all(domain_nd.lower .== 0.0)
        @test all(domain_nd.upper .== 1.0)
    end
end

# Dead-reckon poses in SE(2)
# Script based off
# https://github.com/UMich-CURLY-teaching/UMich-ROB-530-public/blob/main/code-examples/MATLAB/matrix_groups/odometry_propagation_se2.m
#
# Amro Al-Baali

using LinearAlgebra
using SparseArrays

# Sampling period
dt = 0.6;

# Ground truth
x_true = 0:dt:4;
y_true = 0.1 * exp.(0.6 * collect(x_true)) .- 0.1;
θ_true = map(i->atan(y_true[i] - y_true[i-1], x_true[i] - x_true[i-1]), 2:length(x_true));
pushfirst!(θ_true, 0);

# Number of poses
num_poses = length(x_true);

# Noise free control inputs
u_mat_true = [diff(x_true)'; diff(y_true)'; diff(θ_true)'];

# Convert to array of control inputs
u_arr = mapslices(u_i->[u_i], u_mat_true, dims=1);

# Number of particles in the simulation
num_particles = 1000;

# First order covariance propagation around the mean
T_cov_fo = zeros(3, 3);

# Initialize all robot poses
T_all = map(x->I(3), x_true);

# Process model noise
Σ_w = Diagonal([0.03, 0.03, 0.1].^2);

# Cholesky factor for covaraince sampling
L_w = cholesky(Q).L;

# Generators
G₁ = sparse([1], [3], 1, 3, 3);
G₂ = sparse([2], [3], 1, 3, 3);
G₃ = sparse([1, 2], [2, 1], [-1, 1], 3, 3);
# Array of Generators
G = [G₁, G₂, G₃];

# se2 wedge operator
wedge(ξ::Vector{Float64}) = sum(ξ .* G);

# Adjoint function
Ad(X) = [X[1:2, 1:2] [X[2, 3]; -X[1, 3]]; [0 0 1]];

# Noise-free trajectory
poses_true = Array{Float64, 2}[];
push!(poses_true, I(3));
for k = 2:num_poses
  U_km1 = wedge(dt * u_arr[k - 1]);  
  Ξ_km1 = exp(collect(U_km1));
  push!(poses_true, poses_true[k - 1] * Ξ_km1);
end

# Temporar
  

# # Array of noise-free trajectories
# trajs = Vector{Array{Float64, 2}}[];

# for i = 1:num_particles
#   push!(trajs, [I(3)])
# end

# Covariance on latest pose
Σ_xi
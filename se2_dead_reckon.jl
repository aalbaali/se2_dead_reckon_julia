# Dead-reckon poses in SE(2)
# Script based off
# https://github.com/UMich-CURLY-teaching/UMich-ROB-530-public/blob/main/code-examples/MATLAB/matrix_groups/odometry_propagation_se2.m
#
# Amro Al-Baali

using LinearAlgebra
using SparseArrays
using Plots
using Distributions
using Colors
using LaTeXStrings

# Sampling period
dt = 0.5;

# Ground truth
x_true = 0:dt:5;
y_true = 0.1 * exp.(0.6 * collect(x_true)) .- 0.1;
θ_true = map(i -> atan(y_true[i] - y_true[i-1], x_true[i] - x_true[i-1]), 2:length(x_true));
pushfirst!(θ_true, 0);

# Number of poses
num_poses = length(x_true);

# Noise free control inputs
u_mat_true = [diff(x_true)'; diff(y_true)'; diff(θ_true)'];

# Convert to array of control inputs
u_arr = mapslices(u_i -> [u_i], u_mat_true, dims = 1);

# Number of particles in the simulation
num_particles = 1000;

# First order covariance propagation around the mean
T_cov_fo = zeros(3, 3);

# Process model noise
Σ_w = Diagonal([0.03, 0.03, 0.1] .^ 2);

# Cholesky factor for covaraince sampling
L_w = cholesky(Σ_w).L;

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
traj_true = Array{Float64,2}[];
push!(traj_true, I(3));
for k = 2:num_poses
    U_km1 = wedge(dt * u_arr[k-1])
    Ξ_km1 = exp(collect(U_km1))
    push!(traj_true, traj_true[k-1] * Ξ_km1)
end

# Noisy trajectories
trajectories = Vector{Array{Float64}}[];

# Initialize trajectories
for i = 1:num_particles
    push!(trajectories, [I(3)])
end

# State covariance of last pose
Σ_xi = 1e-7 * I(3);

# Go over the trajectory
for k = 2:num_poses
    for p = 1:num_particles
        # Get noisy measurement
        u_km1 = u_arr[k-1] + L_w * randn(3)
        Ξ_km1 = exp(collect(wedge(dt * u_km1)))

        T_km1 = trajectories[p][k-1]
        push!(trajectories[p], T_km1 * Ξ_km1)

        # Propagate covariance of last pose
        if p == num_particles
            # Jacobian of process model w.r.t. T_km1 (pose)
            A = Ad(inv(Ξ_km1))

            # Jacobian of process model w.r.t. process noise          
            L = dt * I(3)

            # Update covariance
            global Σ_xi = Symmetric(A * Σ_xi * A' + L * Σ_w * L')
        end
    end
end

# Generate plots
plot_font = "Computer Modern"
default(fontfamily = plot_font, linewidth = 2, grid = true, thickness_scaling = 1)
scalefontsizes();

traj_x = map(T -> T[1, 3], traj_true);
traj_y = map(T -> T[2, 3], traj_true);
p = plot(traj_x, traj_y, label = "Trajectory");
xlabel!("x [m]");
ylabel!("y [m]");
display(p);

# Get the points of the last pose from all particles
x = map(trajs -> trajs[end][1, 3], trajectories);
y = map(trajs -> trajs[end][2, 3], trajectories);
plt_scatter = scatter!(x, y, aspect_ratio = :equal, label = L"\mathbf{T}^{i}_{K}");
display(plt_scatter)

# Display uncertainty bounds
ϕ = -π:0.01:π;
circle = [cos.(ϕ) sin.(ϕ) zeros(length(ϕ))];
α = 0.001;
scale = sqrt(quantile(Chisq(3), 1 - α));

# Covariance ellipse on manifold
# Get TRUE last pose
T_end = traj_true[end];
ellipse = Vector{Float64}[];
# Lower cholesky factor
L_xi = cholesky(Σ_xi).L;
for p = 1:length(ϕ)
    ell_se2 = scale * L_xi * circle[p, :]
    T_pt = T_end * exp(collect(wedge(ell_se2)))
    push!(ellipse, T_pt[1:2, 3])
end

x_ellipse = map(v -> v[1], ellipse);
y_ellipse = map(v -> v[2], ellipse);
p2 = plot!(x_ellipse, y_ellipse, label = "$((1-α)*100)% confidence bounds",
            legend = :topleft);
display(p2);

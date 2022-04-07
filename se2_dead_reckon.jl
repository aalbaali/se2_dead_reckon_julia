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

################################################################################
# Sim paramaters
################################################################################
# Sampling period
dt = 0.5;

# Sim stopping time
t_end = 15;

# Flag for using left/right Jacobians for computing the uncertainty bounds
use_manifold_jacobians = true;

# Use noise on group rather than Lie algebra (i.e., TₖExp(uₖ)Exp(wₖ) instead of TₖExp(uₖ + wₖ))
noise_on_manifold = false;

# Number of particles in the simulation
num_particles = 1000;

# Process model noise
Σ_w = Diagonal([0.03, 0.03, 0.1] .^ 2);

# Covaraince on initial state
Σ_xi_0 = 1e-7 * I(3);

# p-value for uncertainty bounds
α = 0.001;

# Plotting params
plot_font = "Computer Modern"
default(fontfamily = plot_font, linewidth = 2, grid = true, thickness_scaling = 1)

# Generate a new plot if this is true
should_create_new_plot = true;

################################################################################
# Constants
################################################################################
# Generators
G₁ = sparse([1], [3], 1, 3, 3);
G₂ = sparse([2], [3], 1, 3, 3);
G₃ = sparse([1, 2], [2, 1], [-1, 1], 3, 3);

# Array of Generators
G = [G₁, G₂, G₃];

################################################################################
# Supporting functios
################################################################################
function wraptoπ(θ::Real)
    return (θ + π) % (2π) - π
end

# se2 wedge operator
function wedge(ξ::Vector{Real})
    return sum(ξ .* G)
end

# Adjoint function
function Ad(X::Matrix)
    return [X[1:2, 1:2] [X[2, 3]; -X[1, 3]]; [0 0 1]]
end

# Y as a function of x
function ytraj(x::Real)
    return 0.1 * exp(0.6 * x) - 0.1
end

"""
Retract/map covariance from Lie algebra onto the manifold
"""
function retractCovEllipse(T, Σ, α::Real, num_points=100)
    ϕ = LinRange(-π, π, num_points);
    circle = [cos.(ϕ) sin.(ϕ) zeros(length(ϕ))];
    scale = sqrt(quantile(Chisq(3), 1 - α));
    ellipse = Vector{Float64}[];

    # Lower cholesky factor
    L_xi = cholesky(Σ).L;
    for p = 1:length(ϕ)
        ell_se2 = scale * L_xi * circle[p, :]
        T_pt = T * exp(collect(wedge(ell_se2)))
        push!(ellipse, T_pt[1:2, 3])
    end

    return ellipse;
end

################################################################################
# Main code
################################################################################
# Ground truth
x_true = 0:dt:t_end;
y_true = ytraj.(x_true);
θ_true = map(i -> atan(y_true[i] - y_true[i-1], x_true[i] - x_true[i-1]), 2:length(x_true));
pushfirst!(θ_true, 0);

# Number of poses
num_poses = length(x_true);

# Noise free control inputs
u_mat_true = [diff(x_true)'; diff(y_true)'; diff(θ_true)'];

# Convert to array of control inputs
u_arr = mapslices(u_i -> [u_i], u_mat_true, dims = 1);

# Cholesky factor for covaraince sampling
L_w = cholesky(Σ_w).L;

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
Σ_xi = Σ_xi_0;

# Go over the trajectory
for k = 2:num_poses
    for p = 1:num_particles
        # Get noisy measurement
        w_km1 = L_w * randn(3)
        u_km1 = u_arr[k-1];
        if noise_on_manifold
            W_km1 = exp(collect(wedge(w_km1)));
        else
            global u_km1 += w_km1;
        end
        Ξ_km1 = exp(collect(wedge(dt * u_km1)))


        T_km1 = trajectories[p][k-1]
        if noise_on_manifold
            push!(trajectories[p], T_km1 * Ξ_km1 * W_km1);
        else
            push!(trajectories[p], T_km1 * Ξ_km1);
        end

        # Propagate covariance of last pose
        if p == num_particles
            # Jacobian of process model w.r.t. T_km1 (pose)
            local A = Ad(inv(Ξ_km1))

            # Jacobian of process model w.r.t. process noise
            if use_manifold_jacobians
                # Using (68) and (163) from Sola
                local θᵤ = wraptoπ(u_km1[3])
                if θᵤ ≈ 0
                    Jᵣ = I(3);
                else
                    J_so2_r = [sin(θᵤ)/θᵤ        (1-cos(θᵤ))/θᵤ;
                               (cos(θᵤ)-1)/θᵤ    sin(θᵤ)/θᵤ];
                    J_ρ_r = [(θᵤ * u_km1[1]-u_km1[2]+u_km1[2]cos(θᵤ)-u_km1[1]sin(θᵤ)) / θᵤ^2;
                             (-u_km1[1] + θᵤ * u_km1[2] + u_km1[1]cos(θᵤ) - u_km1[2]sin(θᵤ)) / θᵤ^2];
                    Jᵣ = [J_so2_r J_ρ_r;
                          0 0 1];
                end
                L = dt * Jᵣ;
            else
                L = dt * I(3);
            end

            # Update covariance
            global Σ_xi = Symmetric(A * Σ_xi * A' + L * Σ_w * L')
        end
    end
end

traj_x = map(T -> T[1, 3], traj_true);
traj_y = map(T -> T[2, 3], traj_true);
p = plot!(traj_x, traj_y, label = "Trajectory");
xlabel!("x [m]");
ylabel!("y [m]");
display(p);

# Get the points of the last pose from all particles
x = map(trajs -> trajs[end][1, 3], trajectories);
y = map(trajs -> trajs[end][2, 3], trajectories);
plt_scatter = scatter!(x, y, aspect_ratio = :equal, label = L"\mathbf{T}^{i}_{K}");
display(plt_scatter)

ellipse = retractCovEllipse(traj_true[end], Σ_xi, α, 100);

x_ellipse = map(v -> v[1], ellipse);
y_ellipse = map(v -> v[2], ellipse);
p2 = plot!(
    x_ellipse,
    y_ellipse,
    label = "$((1-α)*100)% confidence bounds",
    legend = :bottomleft,
);
display(p2);

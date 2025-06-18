#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor (Student Version)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    """
    Solve 2D Laplace equation using SOR method for finite thickness parallel plate capacitor.
    
    Args:
        nx (int): Number of grid points in x direction
        ny (int): Number of grid points in y direction
        plate_thickness (int): Thickness of conductor plates in grid points
        plate_separation (int): Separation between plates in grid points
        omega (float): Relaxation factor (1.0 < omega < 2.0)
        max_iter (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        
    Returns:
        np.ndarray: 2D electric potential distribution
    """
    # Initialize potential grid
    potential = np.zeros((ny, nx))
    
    # Calculate plate positions
    mid_y = ny // 2
    upper_plate_top = mid_y + plate_separation // 2
    upper_plate_bottom = upper_plate_top - plate_thickness
    lower_plate_top = mid_y - plate_separation // 2
    lower_plate_bottom = lower_plate_top - plate_thickness
    
    # Set boundary conditions
    # Upper plate (+100V)
    potential[upper_plate_bottom:upper_plate_top+1, :] = 100
    # Lower plate (-100V)
    potential[lower_plate_bottom:lower_plate_top+1, :] = -100
    # Left and right boundaries (0V)
    potential[:, 0] = 0
    potential[:, -1] = 0
    
    # Create mask for fixed potential points (plates and boundaries)
    fixed_mask = np.zeros_like(potential, dtype=bool)
    fixed_mask[upper_plate_bottom:upper_plate_top+1, :] = True
    fixed_mask[lower_plate_bottom:lower_plate_top+1, :] = True
    fixed_mask[:, 0] = True
    fixed_mask[:, -1] = True
    
    # SOR iteration
    for iteration in range(max_iter):
        max_diff = 0.0
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if not fixed_mask[i, j]:
                    new_val = (1-omega) * potential[i, j] + omega * 0.25 * (
                        potential[i+1, j] + potential[i-1, j] + 
                        potential[i, j+1] + potential[i, j-1])
                    diff = abs(new_val - potential[i, j])
                    if diff > max_diff:
                        max_diff = diff
                    potential[i, j] = new_val
        
        # Check for convergence
        if max_diff < tolerance:
            print(f"Converged after {iteration} iterations")
            break
    
    return potential

def calculate_charge_density(potential_grid, dx, dy):
    """
    Calculate charge density using Poisson equation.
    
    Args:
        potential_grid (np.ndarray): 2D electric potential distribution
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        
    Returns:
        np.ndarray: 2D charge density distribution
    """
    # Calculate the Laplacian using finite differences
    laplacian = np.zeros_like(potential_grid)
    
    # Central difference for interior points
    laplacian[1:-1, 1:-1] = (
        (potential_grid[1:-1, 2:] - 2*potential_grid[1:-1, 1:-1] + potential_grid[1:-1, :-2]) / dx**2 +
        (potential_grid[2:, 1:-1] - 2*potential_grid[1:-1, 1:-1] + potential_grid[:-2, 1:-1]) / dy**2
    )
    
    # Calculate charge density from Poisson equation
    charge_density = -laplacian / (4 * np.pi)
    
    return charge_density

def plot_results(potential, charge_density, x_coords, y_coords):
    """
    Create visualization of potential and charge density distributions.
    
    Args:
        potential (np.ndarray): 2D electric potential distribution
        charge_density (np.ndarray): Charge density distribution
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array
    """
    plt.figure(figsize=(12, 10))
    
    # Plot potential distribution
    plt.subplot(2, 2, 1)
    contour = plt.contourf(x_coords, y_coords, potential, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Potential (V)')
    plt.title('Electric Potential Distribution')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    # Plot electric field lines
    plt.subplot(2, 2, 2)
    Ey, Ex = np.gradient(-potential)
    plt.streamplot(x_coords, y_coords, Ex, Ey, color='black', density=1.5)
    plt.title('Electric Field Lines')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    # Plot charge density
    plt.subplot(2, 1, 2)
    charge_contour = plt.contourf(x_coords, y_coords, charge_density, levels=20, cmap='plasma')
    plt.colorbar(charge_contour, label='Charge Density (C/m²)')
    plt.title('Charge Density Distribution')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Simulation parameters
    nx = 100
    ny = 100
    plate_thickness = 10
    plate_separation = 30
    domain_size = 1.0  # 1 meter in each direction
    
    # Calculate grid spacing
    dx = domain_size / (nx - 1)
    dy = domain_size / (ny - 1)
    
    # Solve for potential
    potential = solve_laplace_sor(nx, ny, plate_thickness, plate_separation)
    
    # Calculate charge density
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # Create coordinate arrays for plotting
    x_coords = np.linspace(0, domain_size, nx)
    y_coords = np.linspace(0, domain_size, ny)
    
    # Plot results
    plot_results(potential, charge_density, x_coords, y_coords)

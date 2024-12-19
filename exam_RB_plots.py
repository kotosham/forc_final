import numpy as np
from simulator import Simulator
import pinocchio as pin
from pathlib import Path
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
model = pin.buildModelFromMJCF(xml_path)
data = model.createData()
q0 = np.array([-1.4, -1.3, 1., 0, 0, 0])

def plot_results(times: np.ndarray, positions: np.ndarray, control_inputs: np.ndarray):
    """Plot and save simulation results."""
    error_positions = positions - q0
   
    plt.figure(figsize=(10, 6))
    for i in range(error_positions.shape[1]):
        plt.plot(times, error_positions[:, i], label=f'Error Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Position Error [rad]')
    plt.title('Joint Position Error over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/exam_RB_errors.png')
    plt.close()

    # Plot control inputs
    plt.figure(figsize=(10, 6))
    for i in range(control_inputs.shape[1]):
        plt.plot(times, control_inputs[:, i], label=f'Control Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Input [Nm]')
    plt.title('Control Inputs over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/exam_RB_controls.png')
    plt.close()

def robust_controller(q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:
    """Joint space ID controller.
    
    Args:
        q: Current joint positions [rad]
        dq: Current joint velocities [rad/s]
        t: Current simulation time [s]
        
    Returns:
        tau: Joint torques command [Nm]
    """
    pin.computeAllTerms(model, data, q, dq)
    M = data.M
    nle = data.nle

    # Control gains tuned for UR5e
    k = 2e6
    L = np.diag([200, 200, 200, 100, 100, 100])
    
    # Desired joint configuration
    q0 = np.array([-1.4, -1.3, 1., 0, 0, 0])

    q_wave = q0 - q
    dq_wave = - dq

    s = dq_wave + L @ q_wave

    sigma_max = np.max(np.linalg.svd(np.linalg.inv(M),compute_uv=False))

    V_s = k * s / (np.linalg.norm(s) * sigma_max)
    V = L @ dq_wave + V_s

    # ID control law
    tau = M @ V + nle

    return tau

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    Path("logs/plots").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning real-time joint space control...")

    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        record_video=True,
        enable_task_space=False,
        show_viewer=True,
        video_path="logs/videos/exam_RB_plots.mp4",
        width=1920,
        height=1080
    )

    # Set joint damping (example values, adjust as needed)
    damping = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm/rad/s
    sim.set_joint_damping(damping)
    
    # Set joint friction (example values, adjust as needed)
    friction = np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nm
    sim.set_joint_friction(friction)
    
    # Get original properties
    ee_name = "end_effector"
    
    original_props = sim.get_body_properties(ee_name)
    print(f"\nOriginal end-effector properties:")
    print(f"Mass: {original_props['mass']:.3f} kg")
    print(f"Inertia:\n{original_props['inertia']}")
    
    # Add the end-effector mass and inertia
    sim.modify_body_properties(ee_name, mass=3)
    # Print modified properties
    props = sim.get_body_properties(ee_name)
    print(f"\nModified end-effector properties:")
    print(f"Mass: {props['mass']:.3f} kg")
    print(f"Inertia:\n{props['inertia']}")

    sim.set_controller(robust_controller)
    sim.reset()

    # Simulation parameters
    t = 0
    dt = sim.dt
    time_limit = 5.0
    
    # Data collection
    times = []
    positions = []
    velocities = []
    control_inputs = []
    
    while t < time_limit:
        state = sim.get_state()
        times.append(t)
        positions.append(state['q'])
        velocities.append(state['dq'])
        
        tau = robust_controller(q=state['q'], dq=state['dq'], t=t)
        control_inputs.append(tau)
        sim.step(tau)
        
        if sim.record_video and len(sim.frames) < sim.fps * t:
            sim.frames.append(sim._capture_frame())
        t += dt
    
    # Process and save results
    times = np.array(times)
    positions = np.array(positions)
    control_inputs = np.array(control_inputs)
    
    print(f"Simulation completed: {len(times)} steps")
    print(f"Final joint positions: {positions[-1]}")

    print(positions)
    
    sim._save_video()
    plot_results(times, positions, control_inputs)

if __name__ == "__main__":
    main()
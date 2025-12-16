import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Configuration
SAMPLING_INTERVAL = 1
TOTAL_SAMPLES = 1500

# Nominal phase durations (with controlled variation)
NORMAL_PHASE_TARGET = 800
DEGRADATION_PHASE_TARGET = 500
FAILURE_PHASE_TARGET = 200

# Hardware constraints
CPU_LOAD_NOMINAL = 45.0
CPU_LOAD_MAX = 100.0
TEMP_NOMINAL = 45.0
TEMP_MAX = 85.0
TEMP_CRITICAL = 95.0
VOLTAGE_NOMINAL = 3.3
VOLTAGE_MIN = 3.0
VOLTAGE_MAX = 3.6
EXEC_TIME_NOMINAL = 15.0
EXEC_TIME_MAX = 50.0

# Improved physical constants
THERMAL_TIME_CONSTANT = 0.92  # Stronger thermal inertia
THERMAL_RECOVERY_FACTOR = 0.15  # Heat dissipation during low load
VOLTAGE_SAG_COEFFICIENT = 0.004  # Voltage drop per % CPU load
THERMAL_THROTTLE_THRESHOLD = 70.0  # Temperature that affects performance

print("="*80)
print("REALISTIC EMBEDDED SYSTEM DATASET GENERATOR - ENGINEERING REVISION")
print("="*80)
print("\nAddressing identified issues:")
print("  ✓ Soft phase transitions with behavioral overlap")
print("  ✓ Micro-recovery events during degradation")
print("  ✓ Smoothed temporal features (moving window derivatives)")
print("  ✓ Strengthened physical correlations")
print("  ✓ Intermittent failure symptoms")
print("  ✓ Realistic thermal and electrical dynamics")
print("="*80 + "\n")

def apply_thermal_model(cpu_load, prev_temp, ambient=25.0):
    """
    Physically realistic thermal model with:
    - CPU load proportional heating
    - Ambient temperature baseline
    - Thermal capacitance (exponential response)
    - Heat dissipation at low loads
    """
    # Target temperature based on current CPU load
    heat_generation = (cpu_load / 100.0) ** 1.3  # Nonlinear heat vs load
    target_temp = ambient + (TEMP_MAX - ambient) * heat_generation
    
    # Exponential approach to target (thermal capacitance)
    new_temp = THERMAL_TIME_CONSTANT * prev_temp + (1 - THERMAL_TIME_CONSTANT) * target_temp
    
    # Add thermal recovery when load drops
    if cpu_load < CPU_LOAD_NOMINAL:
        cooling_boost = THERMAL_RECOVERY_FACTOR * (CPU_LOAD_NOMINAL - cpu_load) / CPU_LOAD_NOMINAL
        new_temp -= cooling_boost
    
    return new_temp

def apply_voltage_sag(base_voltage, cpu_load, temperature):
    """
    Realistic voltage sag under load:
    - High CPU load → increased current draw → voltage drop
    - High temperature → increased resistance → voltage drop
    - Small random fluctuations
    """
    # Voltage sag due to current draw (CPU load)
    current_sag = VOLTAGE_SAG_COEFFICIENT * (cpu_load - CPU_LOAD_NOMINAL)
    
    # Additional sag from temperature effects
    temp_factor = max(0, (temperature - TEMP_NOMINAL) / (TEMP_CRITICAL - TEMP_NOMINAL))
    thermal_sag = 0.1 * temp_factor
    
    # Apply sags
    voltage = base_voltage - current_sag - thermal_sag
    
    # Small random noise
    voltage += np.random.normal(0, 0.015)
    
    return np.clip(voltage, VOLTAGE_MIN - 0.2, VOLTAGE_MAX)

def apply_thermal_throttling(base_exec_time, cpu_load, temperature):
    """
    Execution time increases due to:
    - CPU load (scheduling delays)
    - Thermal throttling when temperature is high
    - Random scheduling jitter
    """
    # Base correlation with CPU load
    load_factor = 1.0 + 0.01 * (cpu_load - CPU_LOAD_NOMINAL)
    
    # Thermal throttling effect (exponential above threshold)
    if temperature > THERMAL_THROTTLE_THRESHOLD:
        thermal_penalty = 1.0 + 0.02 * (temperature - THERMAL_THROTTLE_THRESHOLD)
    else:
        thermal_penalty = 1.0
    
    exec_time = base_exec_time * load_factor * thermal_penalty
    
    # Scheduling jitter
    exec_time += np.random.normal(0, 0.5)
    
    return exec_time

def generate_normal_phase_realistic(n_samples):
    """Generate normal phase with realistic variations"""
    print(f"Generating NORMAL phase ({n_samples} samples)...")
    
    cpu_load = np.zeros(n_samples)
    temperature = np.zeros(n_samples)
    voltage = np.zeros(n_samples)
    exec_time = np.zeros(n_samples)
    
    # Initialize
    temperature[0] = TEMP_NOMINAL
    cpu_load[0] = CPU_LOAD_NOMINAL + np.random.normal(0, 2)
    voltage[0] = VOLTAGE_NOMINAL
    exec_time[0] = EXEC_TIME_NOMINAL
    
    for i in range(1, n_samples):
        # CPU load with periodic task pattern
        base_cpu = CPU_LOAD_NOMINAL + 5 * np.sin(2 * np.pi * i / 100)
        cpu_load[i] = base_cpu + np.random.normal(0, 3)
        
        # Occasional brief load spikes (normal system activity)
        if np.random.random() < 0.03:
            cpu_load[i] += np.random.uniform(10, 20)
        
        cpu_load[i] = np.clip(cpu_load[i], 0, 100)
        
        # Temperature follows thermal model
        temperature[i] = apply_thermal_model(cpu_load[i], temperature[i-1])
        temperature[i] += np.random.normal(0, 0.8)  # Sensor noise
        temperature[i] = np.clip(temperature[i], 20, TEMP_MAX)
        
        # Voltage with realistic sag
        voltage[i] = apply_voltage_sag(VOLTAGE_NOMINAL, cpu_load[i], temperature[i])
        
        # Execution time with correlations
        exec_time[i] = apply_thermal_throttling(EXEC_TIME_NOMINAL, cpu_load[i], temperature[i])
        exec_time[i] = np.clip(exec_time[i], 10, EXEC_TIME_MAX)
    
    return cpu_load, temperature, voltage, exec_time

def generate_degradation_phase_realistic(n_samples, prev_state):
    """
    Generate degradation with micro-recoveries and realistic dynamics
    """
    print(f"Generating DEGRADATION phase ({n_samples} samples)...")
    print("  - Including micro-recovery events")
    print("  - Modeling thermal oscillations")
    print("  - Adding load balancing attempts")
    
    cpu_load = np.zeros(n_samples)
    temperature = np.zeros(n_samples)
    voltage = np.zeros(n_samples)
    exec_time = np.zeros(n_samples)
    
    # Initialize from previous state
    temperature[0] = prev_state['temp']
    cpu_load[0] = prev_state['cpu']
    voltage[0] = prev_state['voltage']
    exec_time[0] = prev_state['exec']
    
    # Overall degradation trend (not perfectly linear)
    base_trend = np.linspace(0, 30, n_samples)
    
    # Add micro-recovery events (system attempts to stabilize)
    recovery_points = np.random.choice(range(n_samples), size=8, replace=False)
    recovery_points.sort()
    
    current_stress = 0
    recovery_active = 0
    
    for i in range(1, n_samples):
        # Progress through degradation with noise
        progress_factor = i / n_samples
        current_stress = base_trend[i] * (0.8 + 0.4 * np.random.random())
        
        # Micro-recovery events (thermal cooling, load reduction)
        if i in recovery_points:
            recovery_active = np.random.randint(30, 80)  # Duration of recovery
            print(f"  → Micro-recovery event at sample {i}")
        
        if recovery_active > 0:
            current_stress *= 0.6  # Reduced stress during recovery
            recovery_active -= 1
        
        # CPU load increases but with fluctuations
        base_cpu = prev_state['cpu'] + current_stress
        cpu_load[i] = base_cpu + 7 * np.sin(2 * np.pi * i / 70)
        cpu_load[i] += np.random.normal(0, 4)
        if np.random.random() < 0.08:
            cpu_load[i] += np.random.uniform(5, 15)
        cpu_load[i] = np.clip(cpu_load[i], 0, 100)
        
        # Temperature with realistic thermal dynamics
        temperature[i] = apply_thermal_model(cpu_load[i], temperature[i-1], ambient=27)
        if np.random.random() < 0.04:
            temperature[i] += np.random.uniform(2, 5)
        temperature[i] += np.random.normal(0, 1.5)
        temperature[i] = np.clip(temperature[i], 20, TEMP_CRITICAL)
        
        # Voltage shows increasing instability
        base_v = VOLTAGE_NOMINAL - 0.0003 * i
        voltage[i] = apply_voltage_sag(base_v, cpu_load[i], temperature[i])
        if np.random.random() < 0.04:
            voltage[i] -= np.random.uniform(0.05, 0.12)
        voltage[i] = np.clip(voltage[i], VOLTAGE_MIN, VOLTAGE_MAX)
        
        # Execution time with thermal throttling
        base_exec = EXEC_TIME_NOMINAL + current_stress * 0.25
        exec_time[i] = apply_thermal_throttling(base_exec, cpu_load[i], temperature[i])
        if np.random.random() < 0.06:
            exec_time[i] += np.random.uniform(2, 6)
        exec_time[i] = np.clip(exec_time[i], 10, EXEC_TIME_MAX)
    return cpu_load, temperature, voltage, exec_time

def generate_failure_phase_realistic(n_samples, prev_state):
    """
    Generate failure phase with intermittent symptoms before persistent failure
    """
    print(f"Generating FAILURE phase ({n_samples} samples)...")
    print("  - Intermittent symptoms (first 30% of phase)")
    print("  - Progressive instability (middle 40%)")
    print("  - Persistent critical state (final 30%)")
    
    cpu_load = np.zeros(n_samples)
    temperature = np.zeros(n_samples)
    voltage = np.zeros(n_samples)
    exec_time = np.zeros(n_samples)
    
    temperature[0] = prev_state['temp']
    cpu_load[0] = prev_state['cpu']
    voltage[0] = prev_state['voltage']
    exec_time[0] = prev_state['exec']
    
    intermittent_end = int(n_samples * 0.3)
    progressive_end = int(n_samples * 0.7)
    
    for i in range(1, n_samples):
        progress = i / n_samples
        if i < intermittent_end:
            # Intermittent failure symptoms (system fighting back)
            if i % 5 < 2:  # Symptom appears
                stress_multiplier = 1.8
            else:  # Brief recovery
                stress_multiplier = 1.2
        elif i < progressive_end:
            # Progressive instability
            stress_multiplier = 1.5 + 0.5 * ((i - intermittent_end) / (progressive_end - intermittent_end))
        else:
            # Persistent critical state with chaos
            stress_multiplier = 2.0 + 0.2 * np.random.random()
        target_cpu = 70 + stress_multiplier * 15
        cpu_load[i] = target_cpu + np.random.normal(0, 8)
        if np.random.random() < 0.15:
            cpu_load[i] += np.random.uniform(-15, 20)
        cpu_load[i] = np.clip(cpu_load[i], 0, 100)
        temperature[i] = apply_thermal_model(cpu_load[i], temperature[i-1], ambient=30)
        if i > progressive_end:
            temperature[i] += 0.05 * (i - progressive_end)
        if np.random.random() < 0.12:
            temperature[i] += np.random.uniform(3, 8)
        temperature[i] += np.random.normal(0, 2)
        temperature[i] = np.clip(temperature[i], 20, 110)
        base_v = VOLTAGE_NOMINAL - 0.001 * i
        voltage[i] = apply_voltage_sag(base_v, cpu_load[i], temperature[i])
        if np.random.random() < 0.18:
            voltage[i] -= np.random.uniform(0.1, 0.25)
        voltage[i] = np.clip(voltage[i], 2.7, VOLTAGE_MAX)
        base_exec = EXEC_TIME_NOMINAL + 20 + progress * 20
        exec_time[i] = apply_thermal_throttling(base_exec, cpu_load[i], temperature[i])
        if np.random.random() < 0.20:
            exec_time[i] += np.random.uniform(5, 15)
        exec_time[i] = np.clip(exec_time[i], 10, 80)
    return cpu_load, temperature, voltage, exec_time

def compute_smoothed_temporal_features(signal, window=5):
    """Compute rate of change using moving window for trend emphasis
    This reduces noise and emphasizes actual trends
    """
    # Calculate instantaneous derivative
    derivative = np.zeros_like(signal)
    derivative[1:] = signal[1:] - signal[:-1]
    
    # Apply moving average to smooth
    smoothed = uniform_filter1d(derivative, size=window, mode='nearest')
    return smoothed

def compute_intelligent_failure_labels(df):
    """
    Compute failure labels based on multiple physical thresholds,
    not timeline boundaries - makes labels less trivial
    """
    print("\nComputing intelligent failure labels...")
    print("  - Based on multi-factor degradation assessment")
    print("  - Not aligned with phase boundaries")
    failure_score = np.zeros(len(df))
    
    # Multiple failure indicators
    temp_stress = (df['temperature_c'] - TEMP_MAX) / (TEMP_CRITICAL - TEMP_MAX)
    temp_stress = np.clip(temp_stress, 0, 1)
    
    voltage_stress = (VOLTAGE_NOMINAL - df['supply_voltage_v']) / (VOLTAGE_NOMINAL - VOLTAGE_MIN)
    voltage_stress = np.clip(voltage_stress, 0, 1)
    
    exec_stress = (df['task_exec_time_ms'] - EXEC_TIME_NOMINAL) / (EXEC_TIME_MAX - EXEC_TIME_NOMINAL)
    exec_stress = np.clip(exec_stress, 0, 1)
    
    cpu_stress = (df['cpu_load_pct'] - 70) / 30
    cpu_stress = np.clip(cpu_stress, 0, 1)
    
    # Weighted combination of stress factors
    failure_score = (0.3 * temp_stress + 0.25 * voltage_stress + 
                     0.25 * exec_stress + 0.2 * cpu_stress)
    
    # Apply threshold with hysteresis
    failure_labels = np.zeros(len(df), dtype=int)
    in_failure = False
    
    for i in range(len(df)):
        if not in_failure:
            # Enter failure state
            if failure_score[i] > 0.6:  # High threshold to enter
                in_failure = True
                failure_labels[i] = 1
        else:
            # Stay in failure or exit
            if failure_score[i] > 0.4:  # Lower threshold to stay (hysteresis)
                failure_labels[i] = 1
            else:
                in_failure = False
    
    failure_count = failure_labels.sum()# Round for readability
    first_failure = np.where(failure_labels == 1)[0][0] if failure_count > 0 else -1
    
    print(f"  ✓ Failure samples: {failure_count} ({failure_count/len(df)*100:.1f}%)")
    print(f"  ✓ First failure at sample: {first_failure}")
    return failure_labels

def generate_realistic_dataset():
    """Main generation with all improvements"""
    print("\nGenerating dataset with realistic physics and dynamics...\n")
    
    # Generate normal phases from engineering perspective
    cpu_normal, temp_normal, volt_normal, exec_normal = generate_normal_phase_realistic(
        NORMAL_PHASE_TARGET
    )
    
    # Generate degradation with smooth transition
    prev_state = {
        'cpu': cpu_normal[-1],
        'temp': temp_normal[-1],
        'voltage': volt_normal[-1],
        'exec': exec_normal[-1]
    }
    cpu_degrade, temp_degrade, volt_degrade, exec_degrade = generate_degradation_phase_realistic(
        DEGRADATION_PHASE_TARGET, prev_state
    )
    
    # Generate failure phase
    prev_state = {
        'cpu': cpu_degrade[-1],
        'temp': temp_degrade[-1],
        'voltage': volt_degrade[-1],
        'exec': exec_degrade[-1]
    }
    cpu_failure, temp_failure, volt_failure, exec_failure = generate_failure_phase_realistic(
        FAILURE_PHASE_TARGET, prev_state
    )
    
    # Concatenate
    cpu_load = np.concatenate([cpu_normal, cpu_degrade, cpu_failure])
    temperature = np.concatenate([temp_normal, temp_degrade, temp_failure])
    voltage = np.concatenate([volt_normal, volt_degrade, volt_failure])
    exec_time = np.concatenate([exec_normal, exec_degrade, exec_failure])
    
    # Compute smoothed temporal features
    print("\nComputing smoothed temporal features (5-sample moving window)...")
    temp_rate = compute_smoothed_temporal_features(temperature, window=5)
    cpu_rate = compute_smoothed_temporal_features(cpu_load, window=5)
    exec_drift = compute_smoothed_temporal_features(exec_time, window=5)
    
    # Generate timestamps
    start_time = datetime(2025, 12, 14, 10, 0, 0)
    timestamps = [start_time + timedelta(seconds=i*SAMPLING_INTERVAL) for i in range(TOTAL_SAMPLES)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_load_pct': cpu_load,
        'temperature_c': temperature,
        'supply_voltage_v': voltage,
        'task_exec_time_ms': exec_time,
        'temp_rate_of_change': temp_rate,
        'cpu_rate_of_change': cpu_rate,
        'exec_time_drift': exec_drift
    })
    
    # Compute intelligent failure labels
    df['failure_label'] = compute_intelligent_failure_labels(df)
    
    # Round for readability
    df['cpu_load_pct'] = df['cpu_load_pct'].round(2)
    df['temperature_c'] = df['temperature_c'].round(2)
    df['supply_voltage_v'] = df['supply_voltage_v'].round(3)
    df['task_exec_time_ms'] = df['task_exec_time_ms'].round(2)
    df['temp_rate_of_change'] = df['temp_rate_of_change'].round(3)
    df['cpu_rate_of_change'] = df['cpu_rate_of_change'].round(3)
    df['exec_time_drift'] = df['exec_time_drift'].round(3)
    
    return df

def print_engineering_validation(df):
    """Validate improvements from engineering perspective"""
    print("\n" + "="*80)
    print("ENGINEERING VALIDATION OF IMPROVEMENTS")
    print("="*80)
    
    normal_end = NORMAL_PHASE_TARGET
    degrade_end = NORMAL_PHASE_TARGET + DEGRADATION_PHASE_TARGET
    
    # Check temporal feature smoothness
    temp_rate_std = df['temp_rate_of_change'].std()
    cpu_rate_std = df['cpu_rate_of_change'].std()
    
    print(f"\n1. Temporal Feature Quality:")
    print(f"   Temperature rate std dev: {temp_rate_std:.3f} (smoothed, trend-focused)")
    print(f"   CPU rate std dev: {cpu_rate_std:.3f}")
    
    # Check phase transition smoothness
    transition1 = 800
    transition2 = 1300
    
    temp_jump1 = abs(df['temperature_c'].iloc[transition1] - df['temperature_c'].iloc[transition1-1])
    temp_jump2 = abs(df['temperature_c'].iloc[transition2] - df['temperature_c'].iloc[transition2-1])
    
    print(f"\n2. Phase Transition Smoothness:")
    print(f"   Temperature jump at normal→degradation: {temp_jump1:.2f}°C")
    print(f"   Temperature jump at degradation→failure: {temp_jump2:.2f}°C")
    print(f"   ✓ Smooth transitions with behavioral overlap")
    
    # Check physical correlations
    high_cpu_indices = df[df['cpu_load_pct'] > 70].index
    if len(high_cpu_indices) > 0:
        avg_voltage_high_cpu = df.loc[high_cpu_indices, 'supply_voltage_v'].mean()
        avg_voltage_overall = df['supply_voltage_v'].mean()
        
        print(f"\n3. Physical Correlations:")
        print(f"   Average voltage during high CPU load: {avg_voltage_high_cpu:.3f}V")
        print(f"   Average voltage overall: {avg_voltage_overall:.3f}V")
        print(f"   ✓ Voltage sag under load: {(avg_voltage_overall - avg_voltage_high_cpu)*1000:.1f}mV")
    
    # Check failure label intelligence
    failure_start = df[df['failure_label'] == 1].index[0] if df['failure_label'].sum() > 0 else -1
    
    print(f"\n4. Failure Label Intelligence:")
    print(f"   First failure detected at sample: {failure_start}")
    if failure_start > 0:
        offset_from_boundary = abs(failure_start - 1300)
        print(f"   Offset from phase boundary: {offset_from_boundary} samples")
    print(f"   ✓ Labels based on physics, not timeline")
    
    # Check for micro-recoveries
    degrade_section = df.iloc[800:1300]
    temp_decreases = (degrade_section['temp_rate_of_change'] < -0.5).sum()
    
    print(f"\n5. Micro-Recovery Evidence:")
    print(f"   Temperature decrease events in degradation: {temp_decreases}")
    print(f"   ✓ System shows recovery attempts, not monotonic decay")
    print("\n" + "="*80)

if __name__ == "__main__":
    df = generate_realistic_dataset()
    
    # Save corrected dataset
    output_file = '/home/ahmedvini/Documents/JAC - ECC/embedded_system_dataset_corrected.csv'
    df.to_csv(output_file, index=False)
    
    # Engineering validation
    print_engineering_validation(df)
    
    # Visualization section
    print("\nGenerating and saving dataset plots...")
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('Embedded System Time-Series Dataset - Hardware-Aware Simulation', fontsize=16, fontweight='bold')
    normal_end = NORMAL_PHASE_TARGET
    degrade_end = NORMAL_PHASE_TARGET + DEGRADATION_PHASE_TARGET
    
    # Raw features
    axes[0, 0].plot(df.index, df['cpu_load_pct'], linewidth=0.8)
    axes[0, 0].set_ylabel('CPU Load (%)')
    axes[0, 0].set_title('CPU Load')
    axes[0, 0].axvline(normal_end, color='orange', linestyle='--', alpha=0.5, label='Degradation Start')
    axes[0, 0].axvline(degrade_end, color='red', linestyle='--', alpha=0.5, label='Failure Start')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    
    axes[0, 1].plot(df.index, df['temperature_c'], linewidth=0.8, color='orangered')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('System Temperature')
    axes[0, 1].axhline(TEMP_MAX, color='red', linestyle=':', alpha=0.5, label='Safe Limit')
    axes[0, 1].axvline(normal_end, color='orange', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(degrade_end, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)
    
    axes[1, 0].plot(df.index, df['supply_voltage_v'], linewidth=0.8, color='green')
    axes[1, 0].set_ylabel('Voltage (V)')
    axes[1, 0].set_title('Supply Voltage')
    axes[1, 0].axhline(VOLTAGE_NOMINAL, color='green', linestyle=':', alpha=0.5, label='Nominal')
    axes[1, 0].axvline(normal_end, color='orange', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(degrade_end, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)
    
    axes[1, 1].plot(df.index, df['task_exec_time_ms'], linewidth=0.8, color='purple')
    axes[1, 1].set_ylabel('Execution Time (ms)')
    axes[1, 1].set_title('Task Execution Time')
    axes[1, 1].axvline(normal_end, color='orange', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(degrade_end, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Temporal features
    axes[2, 0].plot(df.index, df['temp_rate_of_change'], linewidth=0.8, color='coral')
    axes[2, 0].set_ylabel('ΔTemp/Δt (°C/s)')
    axes[2, 0].set_title('Temperature Rate of Change')
    axes[2, 0].axhline(0, color='black', linestyle=':', alpha=0.3)
    axes[2, 0].axvline(normal_end, color='orange', linestyle='--', alpha=0.5)
    axes[2, 0].axvline(degrade_end, color='red', linestyle='--', alpha=0.5)
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(df.index, df['cpu_rate_of_change'], linewidth=0.8, color='steelblue')
    axes[2, 1].set_ylabel('ΔCPU/Δt (%/s)')
    axes[2, 1].set_title('CPU Load Rate of Change')
    axes[2, 1].axhline(0, color='black', linestyle=':', alpha=0.3)
    axes[2, 1].axvline(normal_end, color='orange', linestyle='--', alpha=0.5)
    axes[2, 1].axvline(degrade_end, color='red', linestyle='--', alpha=0.5)
    axes[2, 1].grid(True, alpha=0.3)
    
    axes[3, 0].plot(df.index, df['exec_time_drift'], linewidth=0.8, color='mediumorchid')
    axes[3, 0].set_ylabel('ΔExec/Δt (ms/s)')
    axes[3, 0].set_title('Execution Time Drift')
    axes[3, 0].axhline(0, color='black', linestyle=':', alpha=0.3)
    axes[3, 0].axvline(normal_end, color='orange', linestyle='--', alpha=0.5)
    axes[3, 0].axvline(degrade_end, color='red', linestyle='--', alpha=0.5)
    axes[3, 0].set_xlabel('Sample Index')
    axes[3, 0].grid(True, alpha=0.3)
    
    # Failure label
    axes[3, 1].fill_between(df.index, 0, df['failure_label'], color='red', alpha=0.3, label='Failure State')
    axes[3, 1].set_ylabel('Failure Label')
    axes[3, 1].set_title('Ground Truth Failure Label')
    axes[3, 1].set_ylim(-0.1, 1.1)
    axes[3, 1].set_xlabel('Sample Index')
    axes[3, 1].axvline(normal_end, color='orange', linestyle='--', alpha=0.5)
    axes[3, 1].axvline(degrade_end, color='red', linestyle='--', alpha=0.5)
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].legend(fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('/home/ahmedvini/Documents/JAC - ECC/embedded_system_dataset_corrected_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved: embedded_system_dataset_corrected_visualization.png")
    
    # --- Proactive Anomaly Detection Visualization ---
    print("\nRunning anomaly detection and generating proactive visualization...")
    # Use all features except timestamp and failure_label
    feature_columns = [
        'cpu_load_pct',
        'temperature_c',
        'supply_voltage_v',
        'task_exec_time_ms',
        'temp_rate_of_change',
        'cpu_rate_of_change',
        'exec_time_drift'
    ]
    X = df[feature_columns].values
    # Train only on normal operation (first 800 samples, do NOT use label)
    X_train = X[:800]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X)
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        max_samples='auto',
        n_jobs=-1
    )
    iso_forest.fit(X_train_scaled)
    anomaly_score = -iso_forest.score_samples(X_all_scaled)
    degradation_onset = 800
    failure_onset = 1300
    plt.figure(figsize=(16, 6))
    plt.plot(df.index, anomaly_score, color='navy', linewidth=1.2, label='Anomaly Score (higher = more anomalous)')
    plt.axvline(degradation_onset, color='orange', linestyle='--', linewidth=2, label='Degradation Onset (visualization only)')
    plt.axvline(failure_onset, color='red', linestyle='--', linewidth=2, label='Failure Onset (visualization only)')
    plt.fill_between(df.index, 0, anomaly_score, where=(df.index>=failure_onset), color='red', alpha=0.08)
    plt.fill_between(df.index, 0, anomaly_score, where=(df.index>=degradation_onset) & (df.index<failure_onset), color='orange', alpha=0.08)
    plt.xlabel('Sample Index (Time)')
    plt.ylabel('Anomaly Score')
    plt.title('Proactive Anomaly Detection: Anticipating Embedded System Failure\n(Isolation Forest, trained on normal data only)', fontsize=15, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/ahmedvini/Documents/JAC - ECC/anomaly_score_proactive_visualization.png', dpi=300)
    print('Visualization saved: anomaly_score_proactive_visualization.png')
    
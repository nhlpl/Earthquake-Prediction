import math
import random
import statistics

# ---------- Golden Ratio Constants ----------
PHI = (1 + math.sqrt(5)) / 2          # 1.618033988749895
ALPHA = 1 / PHI                       # 0.6180339887498949
BETA = PHI                            # 1.618033988749895
RHO = ALPHA                           # correlation factor for fusion
T_REF = 24.0                          # reference time window (hours)
ALERT_THRESHOLD = 0.618               # golden‑ratio threshold for issuing alert

# Species parameters (threshold, baseline anomaly rate, noise level)
SPECIES = {
    'ant':     {'threshold': 0.3, 'base_rate': 0.02, 'noise_std': 0.05, 'false_pos_rate': 0.01},
    'dog':     {'threshold': 0.2, 'base_rate': 0.01, 'noise_std': 0.04, 'false_pos_rate': 0.008},
    'elephant':{'threshold': 0.1, 'base_rate': 0.005,'noise_std': 0.03, 'false_pos_rate': 0.005}
}

# ---------- Synthetic Earthquake Generator ----------
class EarthquakeGenerator:
    """Generates random earthquakes with exponential inter‑event times."""
    def __init__(self, rate_per_day=0.1, min_mag=4.0, max_mag=7.5, seed=42):
        self.rate = rate_per_day
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.rng = random.Random(seed)
        self.next_time = 0.0

    def next_event(self):
        """Return (time_days, magnitude) of next earthquake."""
        dt = self.rng.expovariate(self.rate)
        self.next_time += dt
        mag = self.rng.uniform(self.min_mag, self.max_mag)
        return self.next_time, mag

# ---------- Animal Anomaly Simulator (with realistic noise) ----------
class AnimalAnomalySimulator:
    def __init__(self, species, threshold, base_rate, noise_std, false_pos_rate, seed=42):
        self.species = species
        self.threshold = threshold
        self.base_rate = base_rate
        self.noise_std = noise_std
        self.false_pos_rate = false_pos_rate
        self.rng = random.Random(seed)

    def anomaly_intensity(self, time_to_quake_hours, magnitude):
        """
        Returns anomaly intensity in [0,1].
        time_to_quake_hours: positive if before quake, negative after.
        """
        # No anomaly after the earthquake
        if time_to_quake_hours < 0:
            return 0.0

        # Golden‑ratio signal: increases as quake approaches
        T = max(time_to_quake_hours, 0.01)
        T0 = 6.18  # characteristic time (hours)
        exponent = (T0 / T) ** BETA
        sigmoid = 1.0 - math.exp(-exponent)
        # Magnitude scaling
        mag_factor = max(0.0, min(1.0, (magnitude - 3.0) / 5.0))
        signal = self.base_rate + (1.0 - self.base_rate) * sigmoid * mag_factor

        # Add Gaussian‑like noise (sum of 12 uniforms approximates normal)
        noise = sum(self.rng.uniform(-self.noise_std, self.noise_std) for _ in range(12))
        intensity = signal + noise

        # Random false positives (completely spurious anomalies)
        if self.rng.random() < self.false_pos_rate:
            intensity = self.rng.uniform(0.5, 1.0)

        return max(0.0, min(1.0, intensity))

# ---------- Golden‑Ratio Probability Model ----------
def species_probability(anomaly_intensity, species_threshold, time_window_hours=6):
    """Probability that an anomaly is a true precursor (Eq. 1)."""
    delta = max(0.0, anomaly_intensity - species_threshold)
    x = delta / ALPHA
    term = (x ** BETA) * (time_window_hours / T_REF)
    return 1.0 - math.exp(-term)

def fuse_probabilities(probs, rho=RHO):
    """Bayesian fusion with correlation penalty."""
    if not probs:
        return 0.0
    # Clamp probabilities to avoid log(0)
    safe_probs = [min(max(p, 1e-10), 1.0-1e-10) for p in probs]
    log_sum = sum(math.log(1 - p) for p in safe_probs)
    return 1.0 - math.exp(rho * log_sum)

def expected_magnitude(anomaly_intensity, species_threshold, M0=4.0):
    delta = max(0.0, anomaly_intensity - species_threshold)
    x = delta / ALPHA
    return M0 + (1.0 / BETA) * math.log(1.0 + x)

# ---------- Simulation Main ----------
def run_simulation(duration_days=365):
    """Run simulation for one year, collect forecasts and evaluate."""
    quake_gen = EarthquakeGenerator(rate_per_day=0.1, seed=1)
    animal_sims = {}
    for name, params in SPECIES.items():
        animal_sims[name] = AnimalAnomalySimulator(
            name,
            params['threshold'],
            params['base_rate'],
            params['noise_std'],
            params['false_pos_rate'],
            seed=hash(name) % 10000
        )

    # Timeline in hours
    current_time = 0.0
    end_time = duration_days * 24.0
    next_quake_time, next_quake_mag = quake_gen.next_event()
    next_quake_time *= 24.0  # convert to hours

    forecasts = []   # (time, probability, quake_in_window)
    alerts = []      # (time, magnitude_estimate, probability)

    while current_time < end_time:
        # Next evaluation point (every 6 hours or at quake time)
        next_eval = min(current_time + 6.0, next_quake_time, end_time)

        # If quake occurs exactly at next_eval, evaluate just before
        if next_quake_time <= next_eval and next_quake_time > current_time:
            t_eval = next_quake_time - 0.01
        else:
            t_eval = next_eval

        # Determine if there is a quake within the next 6 hours from t_eval
        quake_in_window = (0 <= (next_quake_time - t_eval) <= 6.0)

        # Compute anomalies at t_eval (use the upcoming quake if within window)
        anomalies = {}
        if quake_in_window:
            t_to_quake = next_quake_time - t_eval
            for name, sim in animal_sims.items():
                intensity = sim.anomaly_intensity(t_to_quake, next_quake_mag)
                anomalies[name] = intensity
        else:
            # No quake within 6h – use a very large time (background)
            for name, sim in animal_sims.items():
                intensity = sim.anomaly_intensity(1000.0, 4.0)
                anomalies[name] = intensity

        # Compute species probabilities
        probs = []
        for name, intensity in anomalies.items():
            thresh = SPECIES[name]['threshold']
            prob = species_probability(intensity, thresh, time_window_hours=6)
            probs.append(prob)
        fused_prob = fuse_probabilities(probs)

        # Estimate magnitude from the strongest anomaly
        best_species = max(anomalies.items(), key=lambda x: x[1])
        est_mag = expected_magnitude(best_species[1], SPECIES[best_species[0]]['threshold'])

        forecasts.append((t_eval, fused_prob, quake_in_window))
        if fused_prob > ALERT_THRESHOLD:
            alerts.append((t_eval, est_mag, fused_prob))

        # Advance time
        if next_quake_time <= next_eval:
            # Quake occurred, move past it and generate next
            current_time = next_quake_time + 0.01
            next_quake_time, next_quake_mag = quake_gen.next_event()
            next_quake_time *= 24.0
        else:
            current_time = next_eval

    # ---------- Evaluation ----------
    print(f"Simulation over {duration_days} days")
    print(f"Total forecasts: {len(forecasts)}")
    print(f"Number of alerts (probability > {ALERT_THRESHOLD}): {len(alerts)}")
    if alerts:
        avg_est_mag = statistics.mean(mag for _, mag, _ in alerts)
        print(f"Average estimated magnitude on alerts: {avg_est_mag:.2f}")

    # Build ROC curve
    thresholds = [i/20.0 for i in range(1, 20)]  # 0.05 to 0.95
    tpr_list = []
    fpr_list = []
    for thresh in thresholds:
        tp = sum(1 for _, p, q in forecasts if p >= thresh and q)
        fn = sum(1 for _, p, q in forecasts if p < thresh and q)
        fp = sum(1 for _, p, q in forecasts if p >= thresh and not q)
        tn = sum(1 for _, p, q in forecasts if p < thresh and not q)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Add mandatory endpoints (0,0) and (1,1)
    fpr_full = [0.0] + fpr_list + [1.0]
    tpr_full = [0.0] + tpr_list + [1.0]

    # Compute AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr_full)):
        auc += (tpr_full[i] + tpr_full[i-1]) * (fpr_full[i] - fpr_full[i-1]) / 2.0
    print(f"Area under ROC curve (AUC): {auc:.3f}")

    # Show sample forecasts
    print("\nSample forecasts (time_h, probability, quake_in_window):")
    for t, p, q in forecasts[:10]:
        print(f"  t={t:.1f}h, P={p:.3f}, quake={q}")

    # Print ROC curve (simplified)
    print("\nROC curve (approx):")
    for i in range(0, len(thresholds), 2):
        print(f"  FPR={fpr_list[i]:.3f} TPR={tpr_list[i]:.3f}")

    return forecasts, alerts

if __name__ == "__main__":
    random.seed(42)
    run_simulation(365)

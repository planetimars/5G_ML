"""Rule-based optimizer that converts predicted KPIs into actionable strategy plans."""

import numpy as np
import pandas as pd


def _recommended_bandwidth(connected_users):
    # Target operational density around 4 users/MHz.
    raw_needed = connected_users / 4.0
    allowed = np.array([20, 40, 60, 80, 100])
    return int(allowed[np.argmin(np.abs(allowed - raw_needed))])


def generate_optimization_plan(input_df, results_df):
    avg_latency = float(results_df["predicted_latency_ms"].mean())
    avg_throughput = float(results_df["predicted_throughput_mbps"].mean())
    avg_packet_loss = float(input_df["packet_loss"].mean())
    avg_sinr = float(input_df["sinr"].mean())
    avg_jitter = float(input_df["jitter"].mean())
    avg_users = float(input_df["connected_users"].mean())
    avg_bandwidth = float(input_df["bandwidth_mhz"].mean())
    avg_speed = float(input_df["mobility_speed"].mean())

    plan_rows = []

    if avg_latency > 70:
        target_bw = _recommended_bandwidth(avg_users)
        delta_bw = max(0, target_bw - avg_bandwidth)
        action = (
            f"Increase cell bandwidth toward {target_bw} MHz"
            if delta_bw > 0
            else "Rebalance traffic across neighboring cells without bandwidth increase"
        )
        plan_rows.append(
            {
                "Priority": "High",
                "Issue": "Latency above target",
                "Current": f"{avg_latency:.2f} ms",
                "Target": "< 60 ms",
                "PreciseAction": action,
                "EstimatedImpact": "Latency reduction 10-25%",
            }
        )

    if avg_throughput < 80:
        plan_rows.append(
            {
                "Priority": "High",
                "Issue": "Throughput below target",
                "Current": f"{avg_throughput:.2f} Mbps",
                "Target": ">= 100 Mbps",
                "PreciseAction": "Enable scheduler profile favoring high-MCS users and increase PRB allocation by 15-25% during congestion windows",
                "EstimatedImpact": "Throughput gain 12-30%",
            }
        )

    if avg_packet_loss > 1.5:
        plan_rows.append(
            {
                "Priority": "High",
                "Issue": "Packet loss high",
                "Current": f"{avg_packet_loss:.2f}%",
                "Target": "<= 1.0%",
                "PreciseAction": "Apply AQM/queue tuning and raise HARQ retransmission limit by 1 step for edge users",
                "EstimatedImpact": "Packet loss reduction 20-40%",
            }
        )

    if avg_sinr < 12:
        plan_rows.append(
            {
                "Priority": "Medium",
                "Issue": "SINR low",
                "Current": f"{avg_sinr:.2f}",
                "Target": ">= 15",
                "PreciseAction": "Run interference coordination and tilt optimization; prioritize beam refinement for low-SINR sectors",
                "EstimatedImpact": "SINR improvement 2-5 dB",
            }
        )

    if avg_jitter > 8:
        plan_rows.append(
            {
                "Priority": "Medium",
                "Issue": "Jitter elevated",
                "Current": f"{avg_jitter:.2f} ms",
                "Target": "< 5 ms",
                "PreciseAction": "Introduce strict-priority queue for latency-sensitive slices and cap buffer depth for non-critical traffic",
                "EstimatedImpact": "Jitter reduction 15-35%",
            }
        )

    if avg_users > 350:
        plan_rows.append(
            {
                "Priority": "Medium",
                "Issue": "High user density",
                "Current": f"{avg_users:.0f} users",
                "Target": "<= 300 users/cell",
                "PreciseAction": "Enable load balancing/handover bias to distribute 10-20% users to neighboring cells",
                "EstimatedImpact": "Congestion reduction 10-20%",
            }
        )

    if avg_speed > 80:
        plan_rows.append(
            {
                "Priority": "Low",
                "Issue": "High mobility profile",
                "Current": f"{avg_speed:.1f} km/h",
                "Target": "Stable handover KPIs",
                "PreciseAction": "Tune mobility robustness optimization (MRO): decrease Time-to-Trigger and increase HO margin by 1-2 dB",
                "EstimatedImpact": "Fewer handover failures and ping-pong events",
            }
        )

    if not plan_rows:
        plan_rows.append(
            {
                "Priority": "Info",
                "Issue": "Network in healthy zone",
                "Current": "All KPI values near targets",
                "Target": "Maintain baseline",
                "PreciseAction": "Keep current policy, monitor drift weekly, and retrain model monthly",
                "EstimatedImpact": "Sustained QoS stability",
            }
        )

    return pd.DataFrame(plan_rows)

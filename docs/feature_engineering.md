# Feature Engineering

## Initial Feature Set
We began with all extracted flow/DNS features as defined in `feature_menu.yml`.

## Feature Pruning Decisions

### 1. Redundant Linear Combinations
- **Dropped**: `pkts_tot`, `bytes_tot`  
- **Reason**: Both are linear sums of forward and backward values (`fwd + bwd`). Keeping them introduces redundancy.

---

### 2. Highly Correlated Features
We computed Pearson correlations between feature pairs across devices. For correlations > 0.9, we removed the less informative feature.  

- **Packet lengths**  
  - **Kept**: `pktlen_fwd_mean`, `pktlen_fwd_std`, `pktlen_bwd_mean`, `pktlen_bwd_std`  
  - **Dropped**: `pktlen_fwd_min`, `pktlen_fwd_max`, `pktlen_bwd_min`, `pktlen_bwd_max`  
  - **Reason**: mean and std summarize distribution more effectively.  

- **Inter-arrival times (IAT)**  
  - **Kept**: `iat_fwd_mean`, `iat_fwd_std`, `iat_bwd_mean`, `iat_bwd_std`  
  - **Dropped**: `iat_fwd_min`, `iat_fwd_max`, `iat_bwd_min`, `iat_bwd_max`  
  - **Reason**: min/max added little new information beyond mean/std.  

---

### 3. Low Variance / Constant Features
- **Dropped**: `syn_cnt`, `ack_cnt`, `fin_cnt`, `rst_cnt`, `psh_cnt`, `urg_cnt`, `ece_cnt`, `cwr_cnt`, `is_multicast_dst`, `is_broadcast_dst`  
- **Reason**: Almost always zero in our dataset → no discriminative power.  

---

### 4. Overlapping Derived Features
- **Kept**: `down_up_pkt_ratio`  
- **Dropped**: `down_up_byte_ratio` (too correlated with `pktlen` features)  
- **Reason**: ratios were overlapping; we kept the one more interpretable in context.

---

## Final Notes
Feature pruning was guided by:
1. **Correlation analysis** → remove redundant features.  
2. **Variance checks** → remove constant or near-constant features.  
3. **Interpretability** → prefer features that summarize distributions (mean/std) over extreme values (min/max).  

These decisions were made manually, informed by dataset properties, not by an automated feature selection algorithm.

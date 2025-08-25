# Restaurant-Simulation---Spatial-Analysis
This repository contains a restaurant operations simulator built with SimPy + Pygame, designed to analyze how different floor layouts and staffing levels impact service efficiency. The simulator models customers, waiters, and service workflows using Theta* any-angle pathfinding for realistic movement.

### **Description**

This repository contains a **restaurant operations simulator** built with **SimPy + Pygame**, designed to analyze how different floor layouts and staffing levels impact service efficiency. The simulator models customers, waiters, and service workflows using **Theta\*** any-angle pathfinding for realistic movement around obstacles (tables, kitchen, reception).

Key Features:

* Interactive layout design (Entry, Reception, Kitchen, and Tables with groups).
* Configurable parameters: number of waiters, waiter speed, customer arrivals, serving/dwell times.
* Real-time visualization with **grid-based background and labeled coordinates**.
* Detailed event logs and KPI logs (CSV) for analysis.
* KPI metrics include customer wait times, throughput, waiter utilization, and travel distances.
* Supports comparative studies across multiple layouts (Open Hall, Small Bistro, Long Corridor, L-Shaped, Central Kitchen).
* Validated Theta\* travel distances against mathematical ground truth.

This project enables **quantitative evaluation of restaurant layouts** and provides insights into:

* Optimal waiter staffing levels per layout.
* Trade-offs between wait times, throughput, and congestion.
* How geometry (open space vs corridor vs central hub) shapes efficiency.

* # Clone repository
git clone https://github.com/your-username/Restaurant-Simulation---Spatial-Analysis.git
cd Restaurant-Simulation---Spatial-Analysis

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

•	pip install simpy pygame numpy
**Start the app**
•	Exponential: python sim_exp.py (or the filename you use)
•	Deterministic: python sim_det.py




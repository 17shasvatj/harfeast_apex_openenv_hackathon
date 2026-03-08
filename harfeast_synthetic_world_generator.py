"""
HarFeast Synthetic World Generator
Generates all data sources, computes ground truth, and produces task prompts + rubrics
for an APEX-style management consulting RL environment.

Usage:
    python harfeast_world_generator.py [--seed 42] [--output-dir ./world]
"""

import random
import csv
import json
import os
import math
from collections import defaultdict

# =============================================================================
# CONSTANTS
# =============================================================================

PLANTS = [
    "Rockford, Illinois",
    "Madison, Wisconsin",
    "Cedar Rapids, Iowa",
    "Toledo, Ohio",
    "Kalamazoo, Michigan",
]

# Census divisions for labor efficiency calc (Task 5)
PLANT_DIVISIONS = {
    "Rockford, Illinois": "East North Central",
    "Madison, Wisconsin": "East North Central",
    "Cedar Rapids, Iowa": "West North Central",
    "Toledo, Ohio": "East North Central",
    "Kalamazoo, Michigan": "East North Central",
}

ROLES = [
    "Production/Manufacturing Operator",
    "Quality Control/Quality Assurance",
    "Maintenance Technician",
    "Production Supervisor/Team Lead",
    "Supply Chain/Logistics Coordinator",
    "Demand Planning/Forecasting",
    "Administrative/Support Staff",
    "Plant Management",
]

ROLE_TYPES = {
    "Production/Manufacturing Operator": "Front-line",
    "Quality Control/Quality Assurance": "Front-line",
    "Maintenance Technician": "Front-line",
    "Production Supervisor/Team Lead": "Supervisor/Team Lead",
    "Supply Chain/Logistics Coordinator": "Back-office/Support",
    "Demand Planning/Forecasting": "Back-office/Support",
    "Administrative/Support Staff": "Back-office/Support",
    "Plant Management": "Management",
}

PRODUCT_FAMILIES = ["Canned Vegetables", "Condiments", "Sauces"]
EQUIPMENT_TYPES = ["Mixer", "Filler", "Sealer", "Conveyor", "Boiler", "Pasteurizer", "Labeler"]
TRAINING_QUALITY_OPTIONS = [
    "Excellent- comprehensive and very helpful",
    "Good- adequate for most needs",
    "Fair- some gaps or inconsistencies",
    "Poor - insufficient or unhelpful",
]

# Base hourly wages by role (used for wage data file)
BASE_WAGES = {
    "Production/Manufacturing Operator": 18.50,
    "Quality Control/Quality Assurance": 22.00,
    "Maintenance Technician": 25.50,
    "Production Supervisor/Team Lead": 28.00,
    "Supply Chain/Logistics Coordinator": 24.00,
    "Demand Planning/Forecasting": 30.00,
    "Administrative/Support Staff": 20.00,
    "Plant Management": 42.00,
}

# =============================================================================
# DATA GENERATORS
# =============================================================================

def generate_employee_survey(rng, n=3000):
    """Generate the main employee workforce survey dataset."""
    employees = []
    
    # Control distributions to create interesting patterns
    # Toledo and Kalamazoo have higher inefficient hours (for Task 6)
    high_inefficiency_plants = ["Toledo, Ohio", "Kalamazoo, Michigan"]
    # Cedar Rapids and Rockford have highest/lowest willingness (for Task 12)
    
    for i in range(n):
        plant = rng.choice(PLANTS)
        role = rng.choice(ROLES)
        role_type = ROLE_TYPES[role]
        
        # Inefficient hours - higher for Toledo/Kalamazoo
        if plant in high_inefficiency_plants:
            manual = round(rng.uniform(8, 30), 1)
            searching = round(rng.uniform(4, 18), 1)
            fixing = round(rng.uniform(3, 12), 1)
        else:
            manual = round(rng.uniform(0, 8), 1)
            searching = round(rng.uniform(0, 5), 1)
            fixing = round(rng.uniform(0, 4), 1)
        
        # Digital readiness varies by role type
        base_readiness = {"Front-line": 4, "Back-office/Support": 6, 
                          "Supervisor/Team Lead": 5, "Management": 7}
        readiness = round(rng.gauss(base_readiness[role_type], 2), 1)
        readiness = max(1, min(10, readiness))
        
        comfort = round(rng.gauss(5.5, 2), 1)
        comfort = max(1, min(10, comfort))
        
        willing_pilot = rng.choice(["Yes", "No"])
        training_days = rng.choice(["<1 day", "1-2 days", ">2 days"])
        dedicated_time = rng.choice(["Yes", "No"])
        
        training_received = rng.choices(["Yes", "No"], weights=[0.4, 0.6])[0]
        if training_received == "Yes":
            quality = rng.choices(
                TRAINING_QUALITY_OPTIONS,
                weights=[0.16, 0.41, 0.33, 0.10]
            )[0]
        else:
            quality = ""
        
        # Willingness to adopt - slightly higher in Cedar Rapids
        if plant == "Cedar Rapids, Iowa":
            willingness = round(rng.gauss(3.8, 0.8), 1)
        elif plant == "Rockford, Illinois":
            willingness = round(rng.gauss(2.5, 0.8), 1)
        else:
            willingness = round(rng.gauss(3.2, 0.9), 1)
        willingness = max(1, min(5, willingness))
        
        hourly_wage = round(rng.gauss(BASE_WAGES[role], 3), 2)
        hourly_wage = max(12, hourly_wage)
        
        union_status = rng.choice(["Union", "Non-Union"])
        
        employees.append({
            "employee_id": f"EMP-{i:04d}",
            "plant": plant,
            "role": role,
            "role_type": role_type,
            "digital_readiness_score": readiness,
            "digital_comfort_score": comfort,
            "willing_to_pilot": willing_pilot,
            "training_days_willing": training_days,
            "dedicated_training_time": dedicated_time,
            "hours_manual_entry": manual,
            "hours_searching_data": searching,
            "hours_fixing_errors": fixing,
            "hourly_wage": hourly_wage,
            "training_received": training_received,
            "training_quality": quality,
            "willingness_to_adopt": willingness,
            "union_status": union_status,
        })
    
    return employees


def generate_equipment_data(rng):
    """Generate plant equipment dataset. ~50 per plant = 250 total."""
    equipment = []
    eq_id = 0
    
    for plant in PLANTS:
        n_equip = rng.randint(45, 55)
        for j in range(n_equip):
            pf = rng.choice(PRODUCT_FAMILIES)
            et = rng.choice(EQUIPMENT_TYPES)
            
            scheduled = round(rng.uniform(1500, 5000))
            actual = round(scheduled * rng.uniform(0.7, 0.98))
            standard = round(scheduled * rng.uniform(0.85, 1.0))
            labor = round(rng.uniform(500, 3000))
            
            # Scrap rates - most between 3-9%, some outliers
            scrap = round(rng.uniform(0.03, 0.10), 4)
            
            # OEE - plant-dependent base
            oee_base = {"Rockford, Illinois": 0.78, "Madison, Wisconsin": 0.76,
                        "Cedar Rapids, Iowa": 0.80, "Toledo, Ohio": 0.73,
                        "Kalamazoo, Michigan": 0.71}
            oee = round(rng.gauss(oee_base[plant], 0.06), 4)
            oee = max(0.45, min(0.95, oee))
            
            downtime = round(rng.uniform(50, 500))
            units = rng.randint(100000, 600000)
            cogs = round(rng.uniform(800, 2000), 2)
            failure_cost = round(rng.uniform(5000, 50000), 2)
            
            equipment.append({
                "equipment_id": f"EQ-{plant[:3].upper()}-{eq_id:03d}",
                "plant": plant,
                "product_family": pf,
                "equipment_type": et,
                "scheduled_hours": scheduled,
                "actual_hours": actual,
                "standard_hours": standard,
                "labor_hours": labor,
                "scrap_rate": scrap,
                "oee": oee,
                "unplanned_downtime_hours": downtime,
                "units_produced": units,
                "cogs_per_ton": cogs,
                "failure_cost": failure_cost,
            })
            eq_id += 1
    
    return equipment


def generate_quality_losses(rng, equipment):
    """Generate quality losses data derived from equipment data."""
    losses = []
    for eq in equipment:
        scrap_cost = round(eq["cogs_per_ton"] * eq["units_produced"] * eq["scrap_rate"] / 1000, 2)
        failure = round(rng.uniform(2000, 30000), 2)
        losses.append({
            "equipment_id": eq["equipment_id"],
            "plant": eq["plant"],
            "product_family": eq["product_family"],
            "scrap_cost": scrap_cost,
            "unplanned_failure_cost": failure,
        })
    return losses


def generate_plant_labor(rng):
    """Generate per-employee plant labor data for Tasks 5 and 9."""
    labor = []
    lab_id = 0
    
    production_roles = [
        "Production Operator", "Quality Inspector", "Maintenance Tech",
        "Production Supervisor", "Line Lead", "Packaging Operator"
    ]
    
    for plant in PLANTS[:3]:  # Tasks 5 and 9 only use IL, WI, IA plants
        n_workers = rng.randint(15, 25)
        for j in range(n_workers):
            role = rng.choice(production_roles)
            is_supervisor = "Supervisor" in role or "Lead" in role
            wage = round(rng.gauss(22 if is_supervisor else 18, 2), 2)
            wage = max(14, wage)
            
            labor.append({
                "employee_id": f"LAB-{plant[:3].upper()}-{lab_id:03d}",
                "plant": plant,
                "role": role,
                "hourly_wage": wage,
                "annual_hours": 2080,
                "union_status": rng.choice(["Union", "Non-Union"]),
                "supervisor_type": "production" if is_supervisor else "non-production",
                "census_division": PLANT_DIVISIONS[plant],
            })
            lab_id += 1
    
    return labor


def generate_bls_wages():
    """BLS wage benchmark data."""
    return [
        {"occupation": "All Occupations", "industry": "Food Manufacturing", "median_hourly_wage": 19.76},
        {"occupation": "Production Workers", "industry": "Food Manufacturing", "median_hourly_wage": 17.85},
        {"occupation": "Supervisors", "industry": "Food Manufacturing", "median_hourly_wage": 28.50},
        {"occupation": "Maintenance", "industry": "Food Manufacturing", "median_hourly_wage": 24.30},
        {"occupation": "Quality Control", "industry": "Food Manufacturing", "median_hourly_wage": 21.15},
        {"occupation": "Logistics", "industry": "Food Manufacturing", "median_hourly_wage": 22.80},
    ]


def generate_attached_wages():
    """Client-provided updated wage data for Task 10."""
    return [
        {"role": "Production/Manufacturing Operator", "avg_hourly_salary": 21.50},
        {"role": "Quality Control/Quality Assurance", "avg_hourly_salary": 25.80},
        {"role": "Maintenance Technician", "avg_hourly_salary": 29.40},
        {"role": "Production Supervisor/Team Lead", "avg_hourly_salary": 33.20},
        {"role": "Supply Chain/Logistics Coordinator", "avg_hourly_salary": 27.60},
        {"role": "Demand Planning/Forecasting", "avg_hourly_salary": 35.10},
        {"role": "Administrative/Support Staff", "avg_hourly_salary": 23.40},
        {"role": "Plant Management", "avg_hourly_salary": 48.50},
    ]


def generate_oee_assumptions():
    """OEE improvement assumptions for Task 4."""
    return [
        {"plant": "Rockford, Illinois", "current_annual_oee": 0.78, "annual_oee_improvement": 0.030, "investment_start_year": 2025, "world_class_oee_target": 0.85},
        {"plant": "Madison, Wisconsin", "current_annual_oee": 0.76, "annual_oee_improvement": 0.028, "investment_start_year": 2025, "world_class_oee_target": 0.85},
        {"plant": "Cedar Rapids, Iowa", "current_annual_oee": 0.80, "annual_oee_improvement": 0.032, "investment_start_year": 2025, "world_class_oee_target": 0.85},
        {"plant": "Toledo, Ohio", "current_annual_oee": 0.73, "annual_oee_improvement": 0.025, "investment_start_year": 2026, "world_class_oee_target": 0.85},
        {"plant": "Kalamazoo, Michigan", "current_annual_oee": 0.71, "annual_oee_improvement": 0.024, "investment_start_year": 2026, "world_class_oee_target": 0.85},
    ]


def generate_plant_sales():
    """Plant unit sales data for Task 11."""
    return [
        {"plant": "Cedar Rapids, Iowa", "current_unit_sales": 16500000, "price_per_unit": 3.09},
        {"plant": "Rockford, Illinois", "current_unit_sales": 20600000, "price_per_unit": 3.12},
        {"plant": "Madison, Wisconsin", "current_unit_sales": 4680000, "price_per_unit": 5.98},
        {"plant": "Toledo, Ohio", "current_unit_sales": 4890000, "price_per_unit": 6.86},
        {"plant": "Kalamazoo, Michigan", "current_unit_sales": 6400000, "price_per_unit": 6.02},
    ]


def generate_aptean_report():
    """Aptean industry report data for Task 11."""
    return [
        {"technology": "IoT Sensors", "users_growth": 12.5, "non_users_growth": 4.2, "category": "Top Investment to Date"},
        {"technology": "Predictive Maintenance", "users_growth": 11.8, "non_users_growth": 3.9, "category": "Top Planned 2024"},
        {"technology": "Cloud ERP", "users_growth": 9.2, "non_users_growth": 5.1, "category": "Top Investment to Date"},
        {"technology": "Robotic Automation", "users_growth": 10.4, "non_users_growth": 3.5, "category": "Top Planned 2024"},
        {"technology": "AI Quality Control", "users_growth": 8.7, "non_users_growth": 4.8, "category": "Top Investment to Date"},
        {"technology": "Digital Twin", "users_growth": 7.3, "non_users_growth": 4.0, "category": "Other"},
        {"technology": "Supply Chain AI", "users_growth": 6.9, "non_users_growth": 3.2, "category": "Other"},
        {"technology": "Automated Scheduling", "users_growth": 8.1, "non_users_growth": 5.5, "category": "Top Planned 2024"},
        {"technology": "Warehouse Robotics", "users_growth": 7.8, "non_users_growth": 5.0, "category": "Other"},
        {"technology": "Advanced Analytics", "users_growth": 9.8, "non_users_growth": 4.5, "category": "Top Investment to Date"},
    ]


# =============================================================================
# TEXT DOCUMENT GENERATORS
# =============================================================================

def generate_scrap_report():
    return """HarFeast Food Group - Quality Standards: Scrap Rate Report
==========================================================

Acceptable scrap rate range: 4.0% - 7.0%
Target scrap rate (minimum of acceptable range): 4.0%

Plants operating above 7.0% require immediate corrective action and 
must submit a remediation plan within 30 days. Quarterly reviews will 
assess progress toward the target rate.

The target scrap rate represents the minimum of the acceptable range 
and should be used as the baseline for all cost-of-quality calculations.
"""


def generate_interviews():
    interviews = {}
    
    interviews["sarah_jenkins"] = """Expert Interview Transcript - Sarah Jenkins, VP Operations
Date: November 15, 2024

Q: Of the digital levers evaluated, which would deliver the fastest and 
biggest boost to HarFeast's Gross Margin?

A: "We've evaluated several options including predictive maintenance, 
automated scheduling, and IoT-based monitoring. In my assessment, 
IoT Sensors for yield monitoring would deliver the fastest and most 
significant boost to our Gross Margin. The real-time data on production 
yield lets us catch quality issues at the source before they cascade 
into scrap. I've seen it work at comparable food manufacturers with 
measurable margin improvement within 6 months of deployment."
"""
    
    interviews["david_chen"] = """Expert Interview Transcript - David Chen, Director of Manufacturing
Date: November 16, 2024

Q: What digital investment would have the fastest and largest impact 
on HarFeast's profitability?

A: "I've been looking at this from an operations standpoint. While 
predictive maintenance is valuable long-term, the immediate winner 
is IoT Sensors for yield optimization. The ability to monitor yield 
in real-time across all product lines gives us immediate visibility 
into where we're losing margin. Other levers like automated scheduling 
help with throughput but don't directly attack gross margin the way 
yield sensing does. IoT Sensors for yield is my top recommendation."
"""
    
    interviews["mike_russo"] = """Expert Interview Transcript - Mike Russo, Head of Digital Transformation
Date: November 17, 2024

Q: Which digital lever should HarFeast prioritize for the fastest 
margin improvement?

A: "After analyzing all the options, I keep coming back to 
IoT Sensors for yield. The ROI timeline is shortest — typically 
4-8 months to see measurable improvement. Predictive maintenance 
is a close second but has a longer implementation cycle. Cloud ERP 
is foundational but doesn't directly move gross margin in the near 
term. IoT Sensors for yield monitoring is the clear priority if 
we want the fastest and biggest boost to Gross Margin."
"""
    
    return interviews


def generate_frito_lay_case():
    return """Frito-Lay Digital Transformation Case Study
=============================================

Background: Frito-Lay North America, a division of PepsiCo, operates 
over 30 manufacturing facilities producing snack foods including 
Doritos, Cheetos, and Lay's potato chips.

Initiative: In 2022, Frito-Lay deployed IoT-based predictive maintenance 
sensors across their manufacturing network, focusing on high-throughput 
production lines.

Results: After 18 months of deployment, Frito-Lay achieved a 30% 
reduction in unplanned downtime across all monitored production lines. 
The improvement was consistent across facilities regardless of size 
or product type.

Key Success Factors:
- Phased rollout starting with highest-volume lines
- Integration with existing SCADA systems
- Dedicated data analytics team for sensor data interpretation
- Weekly review cadence with plant managers

The 30% unplanned downtime reduction translated to approximately 
$45M in annual cost savings across the network.
"""


def generate_aptean_report_text(aptean_data):
    lines = ["Aptean Food & Beverage Manufacturing Technology Report 2024",
             "=" * 60, "",
             "Top Technology Investments and Revenue Impact Analysis", "",
             f"{'Technology':<25} {'Users Growth':>15} {'Non-Users Growth':>18} {'Category':<28}",
             "-" * 86]
    for row in aptean_data:
        lines.append(f"{row['technology']:<25} {row['users_growth']:>14.1f}% {row['non_users_growth']:>17.1f}% {row['category']:<28}")
    
    lines.extend(["", "",
        "Note: 'Top Investment to Date' and 'Top Planned 2024' represent",
        "investments explicitly identified by surveyed manufacturers as",
        "their highest-priority technology initiatives."])
    return "\n".join(lines)


# =============================================================================
# GROUND TRUTH COMPUTATION
# =============================================================================

def median(values):
    """Compute median of a list of numbers."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def percentile(values, p):
    """Compute percentile using linear interpolation."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0
    k = (n - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def compute_ground_truth(employees, equipment, quality_losses, plant_labor,
                         bls_wages, attached_wages, oee_assumptions,
                         plant_sales, aptean_data):
    """Compute ground truth answers for all 14 tasks."""
    truth = {}
    
    # =========================================================================
    # TASK 1: High-priority employees for digital training rollout
    # =========================================================================
    # Conditions: above role_type median readiness, willing to pilot,
    # >2 days training with dedicated time, above overall median comfort
    
    overall_median_comfort = median([e["digital_comfort_score"] for e in employees])
    
    role_type_readiness_medians = {}
    by_rt = defaultdict(list)
    for e in employees:
        by_rt[e["role_type"]].append(e["digital_readiness_score"])
    for rt, scores in by_rt.items():
        role_type_readiness_medians[rt] = median(scores)
    
    high_priority = []
    for e in employees:
        if (e["digital_readiness_score"] > role_type_readiness_medians[e["role_type"]]
            and e["willing_to_pilot"] == "Yes"
            and e["training_days_willing"] == ">2 days"
            and e["dedicated_training_time"] == "Yes"
            and e["digital_comfort_score"] > overall_median_comfort):
            high_priority.append(e)
    
    hp_count = len(high_priority)
    hp_pct = round(hp_count / len(employees) * 100, 1)
    
    hp_inefficient = sum(e["hours_manual_entry"] + e["hours_searching_data"] + e["hours_fixing_errors"] for e in high_priority)
    total_inefficient = sum(e["hours_manual_entry"] + e["hours_searching_data"] + e["hours_fixing_errors"] for e in employees)
    hp_inefficient_pct = round(hp_inefficient / total_inefficient * 100, 1) if total_inefficient > 0 else 0
    
    hp_by_role_type = defaultdict(int)
    for e in high_priority:
        hp_by_role_type[e["role_type"]] += 1
    
    truth["task1"] = {
        "high_priority_count": hp_count,
        "high_priority_pct": hp_pct,
        "hp_inefficient_hours": round(hp_inefficient, 0),
        "hp_inefficient_pct": hp_inefficient_pct,
        "hp_frontline": hp_by_role_type.get("Front-line", 0),
        "hp_backoffice": hp_by_role_type.get("Back-office/Support", 0),
        "hp_supervisor": hp_by_role_type.get("Supervisor/Team Lead", 0),
        "hp_management": hp_by_role_type.get("Management", 0),
    }
    
    # =========================================================================
    # TASK 2: Adjusted Cost of Instability per plant
    # =========================================================================
    # Formula: Abnormal scrap cost / (Actual Scrap% - Target Scrap%)
    # Target = 4.0% (minimum of acceptable range)
    # Abnormal scrap cost per equipment = COGS_per_ton * units * (scrap_rate - target)
    # Aggregate per plant
    
    target_scrap = 0.04
    
    plant_instability = {}
    for plant in PLANTS:
        plant_equip = [e for e in equipment if e["plant"] == plant]
        total_abnormal_cost = 0
        total_weighted_scrap = 0
        total_units = 0
        
        for eq in plant_equip:
            if eq["scrap_rate"] > target_scrap:
                abnormal = eq["cogs_per_ton"] * eq["units_produced"] * (eq["scrap_rate"] - target_scrap)
                total_abnormal_cost += abnormal
            total_weighted_scrap += eq["scrap_rate"] * eq["units_produced"]
            total_units += eq["units_produced"]
        
        avg_scrap = total_weighted_scrap / total_units if total_units > 0 else 0
        denominator = avg_scrap - target_scrap
        
        if denominator > 0:
            adjusted_cost = round(total_abnormal_cost / denominator)
        else:
            adjusted_cost = 0
        
        plant_instability[plant] = adjusted_cost
    
    truth["task2"] = plant_instability
    
    # =========================================================================
    # TASK 3: Predictive maintenance impact on scrap rate
    # =========================================================================
    # Pilot on equipment where: scheduled_hours >= equipment_type median
    # AND labor_hours >= plant median labor hours
    # Apply 15% scrap reduction to qualifying equipment
    
    # Equipment type median scheduled hours
    by_type = defaultdict(list)
    for eq in equipment:
        by_type[eq["equipment_type"]].append(eq["scheduled_hours"])
    type_median_scheduled = {t: median(hrs) for t, hrs in by_type.items()}
    
    # Plant median labor hours
    by_plant_labor = defaultdict(list)
    for eq in equipment:
        by_plant_labor[eq["plant"]].append(eq["labor_hours"])
    plant_median_labor = {p: median(hrs) for p, hrs in by_plant_labor.items()}
    
    scrap_reduction = 0.15  # 15% reduction for qualifying equipment
    
    # Compute new scrap rates by product family
    pf_data = defaultdict(lambda: {"total_units": 0, "total_scrap_units": 0})
    for eq in equipment:
        qualifies = (eq["scheduled_hours"] >= type_median_scheduled[eq["equipment_type"]]
                     and eq["labor_hours"] >= plant_median_labor[eq["plant"]])
        
        scrap_units = eq["units_produced"] * eq["scrap_rate"]
        if qualifies:
            scrap_units *= (1 - scrap_reduction)
        
        pf_data[eq["product_family"]]["total_units"] += eq["units_produced"]
        pf_data[eq["product_family"]]["total_scrap_units"] += scrap_units
    
    # Also compute original scrap units for avoidance calc
    pf_original = defaultdict(lambda: {"total_units": 0, "total_scrap_units": 0})
    for eq in equipment:
        pf_original[eq["product_family"]]["total_units"] += eq["units_produced"]
        pf_original[eq["product_family"]]["total_scrap_units"] += eq["units_produced"] * eq["scrap_rate"]
    
    task3 = {}
    for pf in PRODUCT_FAMILIES:
        new_rate = round(pf_data[pf]["total_scrap_units"] / pf_data[pf]["total_units"] * 100, 1)
        avoided = round(pf_original[pf]["total_scrap_units"] - pf_data[pf]["total_scrap_units"])
        task3[pf] = {"new_scrap_rate_pct": new_rate, "units_avoided": avoided}
    
    truth["task3"] = task3
    
    # =========================================================================
    # TASK 4: Digital lever agreement + OEE projections
    # =========================================================================
    # Lever is "IoT Sensors for yield" (from interviews)
    # Project OEE per plant until it exceeds world-class target
    
    task4 = {"digital_lever": "IoT Sensors for yield"}
    for oee_row in oee_assumptions:
        plant = oee_row["plant"]
        oee = oee_row["current_annual_oee"]
        improvement = oee_row["annual_oee_improvement"]
        start_year = oee_row["investment_start_year"]
        target = oee_row["world_class_oee_target"]
        
        year = start_year
        while oee < target and year < 2040:
            oee += improvement
            year += 1
        
        if oee >= target:
            task4[plant] = {
                "first_year_exceeds": year,
                "oee_at_that_year": round(oee, 4)
            }
    
    truth["task4"] = task4
    
    # =========================================================================
    # TASK 5: Total labor cost, efficiency gains, union demand
    # =========================================================================
    
    task5 = {}
    for plant in PLANTS[:3]:  # Only IL, WI, IA
        plant_workers = [w for w in plant_labor if w["plant"] == plant]
        
        # Total annual labor cost
        total_cost = sum(w["hourly_wage"] * w["annual_hours"] for w in plant_workers)
        total_cost = round(total_cost)
        
        # Efficiency gains: 10% for West North Central, 20% for others
        # But 5% for non-unionized production supervisors regardless
        efficiency = 0
        for w in plant_workers:
            if w["union_status"] == "Non-Union" and w["supervisor_type"] == "production":
                rate = 0.05
            elif w["census_division"] == "West North Central":
                rate = 0.10
            else:
                rate = 0.20
            efficiency += w["hourly_wage"] * w["annual_hours"] * rate
        efficiency = round(efficiency)
        
        # Union demand: 5% increase for union workers
        union_increase = sum(
            w["hourly_wage"] * w["annual_hours"] * 0.05
            for w in plant_workers if w["union_status"] == "Union"
        )
        union_increase = round(union_increase)
        
        task5[plant] = {
            "total_labor_cost": total_cost,
            "efficiency_gains": efficiency,
            "union_demand_increase": union_increase,
        }
    
    truth["task5"] = task5
    
    # =========================================================================
    # TASK 6: Average inefficient hours per plant
    # =========================================================================
    
    plant_inefficient = defaultdict(list)
    for e in employees:
        total = e["hours_manual_entry"] + e["hours_searching_data"] + e["hours_fixing_errors"]
        plant_inefficient[e["plant"]].append(total)
    
    task6 = {}
    for plant in PLANTS:
        avg = round(sum(plant_inefficient[plant]) / len(plant_inefficient[plant]), 1)
        task6[plant] = avg
    
    sorted_plants = sorted(task6.items(), key=lambda x: x[1])
    most_efficient = [p for p, v in sorted_plants if v == sorted_plants[0][1]]
    least_efficient = sorted_plants[-1][0]
    least_val = sorted_plants[-1][1]
    most_val = sorted_plants[0][1]
    pct_diff = round((least_val - most_val) / most_val * 100)
    
    truth["task6"] = {
        "avg_by_plant": task6,
        "most_efficient": most_efficient,
        "least_efficient": least_efficient,
        "pct_difference": pct_diff,
    }
    
    # =========================================================================
    # TASK 7: Average annual productivity loss per role
    # =========================================================================
    # Survey = 1 week. Annual = multiply by 52.
    
    role_losses = defaultdict(list)
    for e in employees:
        weekly_inefficient = e["hours_manual_entry"] + e["hours_searching_data"] + e["hours_fixing_errors"]
        annual_loss = weekly_inefficient * 52 * e["hourly_wage"]
        role_losses[e["role"]].append(annual_loss)
    
    task7 = {}
    total_annual_loss = 0
    for role in ROLES:
        if role in role_losses:
            avg = round(sum(role_losses[role]) / len(role_losses[role]))
            task7[role] = avg
            total_annual_loss += sum(role_losses[role])
    
    truth["task7"] = {
        "avg_loss_by_role": task7,
        "total_annual_loss": round(total_annual_loss),
    }
    
    # =========================================================================
    # TASK 8: High-priority canned vegetables equipment quality losses
    # =========================================================================
    # High-priority: canned veg with scrap_rate > 5% AND
    # unplanned_downtime_hours > plant median for canned veg
    
    # Plant median downtime for canned vegetables
    cv_by_plant = defaultdict(list)
    for eq in equipment:
        if eq["product_family"] == "Canned Vegetables":
            cv_by_plant[eq["plant"]].append(eq["unplanned_downtime_hours"])
    cv_plant_median = {p: median(hrs) for p, hrs in cv_by_plant.items()}
    
    hp_equip_ids = set()
    for eq in equipment:
        if (eq["product_family"] == "Canned Vegetables"
            and eq["scrap_rate"] > 0.05
            and eq["unplanned_downtime_hours"] > cv_plant_median.get(eq["plant"], 0)):
            hp_equip_ids.add(eq["equipment_id"])
    
    hp_quality_loss = 0
    total_cv_quality_loss = 0
    for ql in quality_losses:
        if ql["product_family"] == "Canned Vegetables":
            loss = ql["scrap_cost"] + ql["unplanned_failure_cost"]
            total_cv_quality_loss += loss
            if ql["equipment_id"] in hp_equip_ids:
                hp_quality_loss += loss
    
    hp_pct_of_cv = round(hp_quality_loss / total_cv_quality_loss * 100) if total_cv_quality_loss > 0 else 0
    
    truth["task8"] = {
        "hp_quality_losses": round(hp_quality_loss),
        "hp_pct_of_cv_losses": hp_pct_of_cv,
    }
    
    # =========================================================================
    # TASK 9: Labor variance for IL and WI plants
    # =========================================================================
    # Variance = Standard Hours - Actual Hours (positive = favorable)
    # Dollar variance = Hours variance * BLS median wage
    # Productivity Index = Actual Hours / Standard Hours
    
    bls_all_occ_wage = next(w["median_hourly_wage"] for w in bls_wages if w["occupation"] == "All Occupations")
    
    task9 = {}
    for plant in ["Rockford, Illinois", "Madison, Wisconsin"]:
        plant_equip = [eq for eq in equipment if eq["plant"] == plant]
        total_standard = sum(eq["standard_hours"] for eq in plant_equip)
        total_actual = sum(eq["actual_hours"] for eq in plant_equip)
        
        variance_hours = round(total_standard - total_actual, 2)
        variance_dollars = round(variance_hours * bls_all_occ_wage, 2)
        productivity_index = round(total_actual / total_standard, 2) if total_standard > 0 else 0
        
        task9[plant] = {
            "variance_hours": variance_hours,
            "variance_dollars": variance_dollars,
            "productivity_index": productivity_index,
        }
    
    truth["task9"] = task9
    
    # =========================================================================
    # TASK 10: Updated productivity loss with attached wages
    # =========================================================================
    # Use attached wage data to get average hourly salary across all roles
    # Then recompute annual productivity loss
    
    avg_hourly = round(sum(w["avg_hourly_salary"] for w in attached_wages) / len(attached_wages), 2)
    
    total_weekly_inefficient = sum(
        e["hours_manual_entry"] + e["hours_searching_data"] + e["hours_fixing_errors"]
        for e in employees
    )
    annual_loss = round(total_weekly_inefficient * 52 * avg_hourly / 1000) * 1000  # in 000s
    
    truth["task10"] = {
        "avg_hourly_wage": avg_hourly,
        "annual_productivity_loss": annual_loss,
    }
    
    # =========================================================================
    # TASK 11: Top 5 tech investments applied to plant sales
    # =========================================================================
    # Filter aptean: only "Top Investment to Date" or "Top Planned 2024"
    # Compute difference: users_growth - non_users_growth
    # Take top 5 by difference
    # Apply cumulative growth to each plant's unit sales
    
    eligible = [a for a in aptean_data if a["category"] in ["Top Investment to Date", "Top Planned 2024"]]
    for a in eligible:
        a["growth_diff"] = a["users_growth"] - a["non_users_growth"]
    eligible.sort(key=lambda x: x["growth_diff"], reverse=True)
    top5 = eligible[:5]
    
    # Total growth multiplier = product of (1 + diff/100) for all 5
    total_growth = 1.0
    for tech in top5:
        total_growth *= (1 + tech["growth_diff"] / 100)
    
    task11 = {}
    for ps in plant_sales:
        new_units = round(ps["current_unit_sales"] * total_growth)
        new_revenue = round(new_units * ps["price_per_unit"])
        task11[ps["plant"]] = {
            "new_unit_sales": new_units,
            "new_projected_sales": new_revenue,
        }
    
    truth["task11"] = {
        "top5_technologies": [t["technology"] for t in top5],
        "plant_results": task11,
    }
    
    # =========================================================================
    # TASK 12: Willingness to adopt by plant and role, training costs
    # =========================================================================
    
    # Plant-level willingness
    plant_willingness = {}
    for plant in PLANTS:
        scores = [e["willingness_to_adopt"] for e in employees if e["plant"] == plant]
        plant_willingness[plant] = round(sum(scores) / len(scores), 2)
    
    sorted_pw = sorted(plant_willingness.items(), key=lambda x: x[1])
    lowest_plant = sorted_pw[0][0]
    highest_plant = sorted_pw[-1][0]
    
    # Role willingness within those plants
    def role_willingness_in_plant(plant):
        by_role = defaultdict(list)
        for e in employees:
            if e["plant"] == plant:
                by_role[e["role"]].append(e["willingness_to_adopt"])
        return {r: round(sum(s)/len(s), 2) for r, s in by_role.items()}
    
    lowest_plant_roles = role_willingness_in_plant(lowest_plant)
    highest_plant_roles = role_willingness_in_plant(highest_plant)
    
    lowest_role_in_lowest = min(lowest_plant_roles.items(), key=lambda x: x[1])
    highest_role_in_highest = max(highest_plant_roles.items(), key=lambda x: x[1])
    
    # Training preferences and costs
    # Preferred training: most common training_days_willing for each role in each plant
    def training_info(plant, role):
        emps = [e for e in employees if e["plant"] == plant and e["role"] == role]
        if not emps:
            return {"preferred_length": "N/A", "count_1_2_days": 0, "total_cost": 0}
        
        prefs = defaultdict(int)
        for e in emps:
            prefs[e["training_days_willing"]] += 1
        
        preferred = max(prefs.items(), key=lambda x: x[1])[0]
        count_1_2 = prefs.get("1-2 days", 0)
        
        # Training cost: $8/hour * hours based on preference
        hours_map = {"<1 day": 4, "1-2 days": 12, ">2 days": 20}
        cost_per_person = 8 * hours_map.get(preferred, 12)  # $8/hr training cost
        total_cost = round(cost_per_person * len(emps))
        
        return {"preferred_length": preferred, "count_1_2_days": count_1_2, "total_cost": total_cost}
    
    truth["task12"] = {
        "lowest_willingness_plant": lowest_plant,
        "highest_willingness_plant": highest_plant,
        "lowest_role_in_lowest_plant": lowest_role_in_lowest,
        "highest_role_in_highest_plant": highest_role_in_highest,
        "training_details": {
            "lowest_plant_lowest_role": training_info(lowest_plant, lowest_role_in_lowest[0]),
            "highest_plant_highest_role": training_info(highest_plant, highest_role_in_highest[0]),
        }
    }
    
    # =========================================================================
    # TASK 13: Apply Frito-Lay downtime reduction
    # =========================================================================
    # 30% reduction in unplanned downtime per plant
    
    task13 = {}
    for plant in PLANTS:
        plant_equip = [eq for eq in equipment if eq["plant"] == plant]
        total_scheduled = sum(eq["scheduled_hours"] for eq in plant_equip)
        total_downtime = sum(eq["unplanned_downtime_hours"] for eq in plant_equip)
        
        current_ratio = total_downtime / total_scheduled if total_scheduled > 0 else 0
        new_ratio = current_ratio * 0.70  # 30% reduction
        task13[plant] = round(new_ratio * 100)  # nearest full percentage point
    
    truth["task13"] = task13
    
    # =========================================================================
    # TASK 14: Training quality breakdown
    # =========================================================================
    
    trained = [e for e in employees if e["training_received"] == "Yes"]
    trained_count = len(trained)
    
    quality_counts = defaultdict(int)
    for e in trained:
        quality_counts[e["training_quality"]] += 1
    
    quality_pcts = {}
    for q in TRAINING_QUALITY_OPTIONS:
        quality_pcts[q] = round(quality_counts[q] / trained_count * 100)
    
    truth["task14"] = {
        "trained_count": trained_count,
        "quality_pcts": quality_pcts,
    }
    
    return truth


# =============================================================================
# TASK PROMPT GENERATION
# =============================================================================

def generate_task_prompts(truth):
    """Generate task prompts adapted to the synthetic world."""
    tasks = []
    
    # TASK 1
    tasks.append({
        "task_id": "task_01",
        "task_name": "High-Priority Digital Training Employees",
        "prompt": """I'm trying to get a sense of which HarFeast employees are most ready for the digital training rollout. Can you pull the workforce survey data and identify all employees who are above their role type's median readiness score, willing to pilot new tools, willing to spend >2 days in training with dedicated training time, and above the overall median digital comfort score?

Once you've identified that group, tell me:
1. How many "high-priority" employees are there, and what % of total employees do they represent?
2. How many total hours does this group spend weekly on manual entry, searching data, or fixing errors? What % of the company-wide total is that?
3. Break down the high-priority count by role type.

Report your answer here.""",
        "ground_truth": truth["task1"],
        "rubric": [
            f"States that the number of high-priority employees is {truth['task1']['high_priority_count']}",
            f"States that the percentage of all employees the high-priority employees represent is {truth['task1']['high_priority_pct']}%",
            f"States that the total hours high-priority employees spend on manual entry, searching data or fixing errors is {truth['task1']['hp_inefficient_hours']:.0f}",
            f"States that the percentage of all such hours from high-priority employees is {truth['task1']['hp_inefficient_pct']}%",
            f"States that the number of high-priority employees in the Front-line role type is {truth['task1']['hp_frontline']}",
            f"States that the number of high-priority employees in the Back-office/Support role type is {truth['task1']['hp_backoffice']}",
            f"States that the number of high-priority employees in the Supervisor/Team Lead role type is {truth['task1']['hp_supervisor']}",
            f"States that the number of high-priority employees in the Management role type is {truth['task1']['hp_management']}",
        ]
    })
    
    # TASK 2
    rubric2 = [f"States that the adjusted cost of instability for {plant} is ${cost:,}" for plant, cost in truth["task2"].items()]
    tasks.append({
        "task_id": "task_02",
        "task_name": "Adjusted Cost of Instability",
        "prompt": """Calculate the Adjusted Cost of Instability for each site, defined as Abnormal scrap cost/(Actual Scrap % - Normal Scrap %) = adjusted cost of instability. The target scrap rate of HarFeast is the minimum in the range of acceptable scrap rate in the scrap rate report. Just use COGS per ton as your scrap cost for now.

Report your final answers to me in a message. Round values to the nearest dollar.""",
        "ground_truth": truth["task2"],
        "rubric": rubric2,
    })
    
    # TASK 3
    rubric3 = []
    for pf in PRODUCT_FAMILIES:
        rubric3.append(f"States that the new overall scrap rate for {pf} is {truth['task3'][pf]['new_scrap_rate_pct']}%")
        rubric3.append(f"States that the scrap units {pf} avoids per year is {truth['task3'][pf]['units_avoided']}")
    tasks.append({
        "task_id": "task_03",
        "task_name": "Predictive Maintenance Scrap Impact",
        "prompt": """Using HarFeast's equipment data, assess the impact of predictive maintenance on HarFeast's scrap rate. We will pilot predictive maintenance only on equipment a) whose scheduled hours per year are at or above that equipment type's median scheduled hours and b) whose labor hours are at or above its plant's median labor hours. For all equipment qualifying for the pilot, apply a 15% reduction to their scrap rate.

Calculate:
1. The new overall scrap rate for each product family (as a %)
2. The total number of scrap units each product family avoids every year

Report rounded to 1 decimal place for rates and nearest whole number for units.""",
        "ground_truth": truth["task3"],
        "rubric": rubric3,
    })
    
    # TASK 4
    rubric4 = [f"States that the digital lever is IoT Sensors for yield"]
    for plant, data in truth["task4"].items():
        if plant == "digital_lever":
            continue
        rubric4.append(f"States that the OEE level for {plant} in the first year exceeding world-class target is {data['oee_at_that_year']:.2%}")
        rubric4.append(f"States that the first year {plant} exceeds world-class target is {data['first_year_exceeds']}")
    tasks.append({
        "task_id": "task_04",
        "task_name": "Digital Lever Agreement and OEE Projections",
        "prompt": """1. What is the digital lever that Sarah Jenkins, David Chen, and Mike Russo agree will deliver the fastest and biggest boost to HarFeast's Gross Margin?

2. Assuming HarFeast adopts the chosen digital lever, determine the OEE level in the first full year in each plant location where the annual OEE value exceeds the world-class target. Use the OEE improvement assumptions file for growth rates and start dates.

Report OEE values to 2 decimal places as percentages.""",
        "ground_truth": truth["task4"],
        "rubric": rubric4,
    })
    
    # TASK 5
    rubric5 = []
    for plant, data in truth["task5"].items():
        rubric5.append(f"States that the Total Annual Labor Cost for {plant} is ${data['total_labor_cost']:,}")
        rubric5.append(f"States that the Efficiency Gains for {plant} is ${data['efficiency_gains']:,}")
        rubric5.append(f"States that the Union Demand Increase for {plant} is ${data['union_demand_increase']:,}")
    tasks.append({
        "task_id": "task_05",
        "task_name": "Labor Cost Analysis",
        "prompt": """1. Give me the total labor cost for each plant location (Rockford IL, Madison WI, Cedar Rapids IA only).

2. Give me the efficiency gains for each plant location. West North Central division plant locations only have a 10% annual efficiency gain from labor cost. For other locations, the efficiency gain is 20%. However, the efficiency gain is 5% for non-unionized production supervisors no matter where they are located.

3. Give me the forecasted labor cost increase from union demands, assuming a 5% increase for all union workers.

Round to the nearest dollar.""",
        "ground_truth": truth["task5"],
        "rubric": rubric5,
    })
    
    # TASK 6
    rubric6 = [f"States the average inefficient time in {plant} is {val}" for plant, val in truth["task6"]["avg_by_plant"].items()]
    for p in truth["task6"]["most_efficient"]:
        rubric6.append(f"States that {p} is a plant with the lowest average inefficient time")
    rubric6.append(f"States that {truth['task6']['least_efficient']} is the plant with the highest average inefficient time")
    rubric6.append(f"States that the difference between highest and lowest average inefficient time is {truth['task6']['pct_difference']}%")
    tasks.append({
        "task_id": "task_06",
        "task_name": "Operational Efficiency Analysis",
        "prompt": """Analyze the operational efficiency at HarFeast and assess how many inefficient employee hours each plant is recording on average. Which plants have the most efficient operations and the least efficient operations? How much more efficient are the highest efficiency locations vs the lowest efficiency locations?

Assume the following activities are considered inefficient: (a) manual data entry, (b) searching for data, (c) fixing errors. Use the workforce survey data. Report averages to 1 decimal place.""",
        "ground_truth": truth["task6"],
        "rubric": rubric6,
    })
    
    # TASK 7
    rubric7 = [f"States the average annual productivity loss cost of a {role} employee is ${loss:,}" for role, loss in truth["task7"]["avg_loss_by_role"].items()]
    rubric7.append(f"States the total annual productivity loss cost is ${truth['task7']['total_annual_loss']:,}")
    tasks.append({
        "task_id": "task_07",
        "task_name": "Productivity Loss Quantification",
        "prompt": """I want to quantify the average annual productivity loss at a cost level for each employee in each primary role based on the sum of average hours spent doing manual entry, searching data, and fixing errors. Then, I want to calculate the total productivity loss cost HarFeast faces every year, company-wide.

Note that the survey responses represent one week of work. Report your final answer as a message. Round to the nearest dollar.""",
        "ground_truth": truth["task7"],
        "rubric": rubric7,
    })
    
    # TASK 8
    tasks.append({
        "task_id": "task_08",
        "task_name": "High-Priority Equipment Quality Losses",
        "prompt": """Using HarFeast's equipment data and quality losses dataset, consider all canned vegetables assets with a scrap rate > 5% and with unplanned downtime hours above the plant median for canned vegetables as "high-priority".

1. For the "high-priority" group, calculate the total annual quality-related losses (scrap cost + unplanned failure cost).
2. What percentage of all canned-vegetable quality losses comes from these high-priority assets?

Report losses rounded to the nearest dollar and percentage to the nearest whole number.""",
        "ground_truth": truth["task8"],
        "rubric": [
            f"States that the total annual quality-related losses for the high-priority group is ${truth['task8']['hp_quality_losses']:,}",
            f"States that the percentage of all canned-vegetable quality losses from high-priority assets is {truth['task8']['hp_pct_of_cv_losses']}%",
        ]
    })
    
    # TASK 9
    rubric9 = []
    for plant, data in truth["task9"].items():
        rubric9.append(f"States that the Labor Efficiency Variance (Hours) for {plant} is {data['variance_hours']} hours")
        rubric9.append(f"States that the Labor Cost Variance for {plant} is ${data['variance_dollars']}")
        rubric9.append(f"States that the Productivity Index for {plant} is {data['productivity_index']}")
    tasks.append({
        "task_id": "task_09",
        "task_name": "Labor Variance Analysis",
        "prompt": """Calculate the total labor variance in hours (favorable should be positive) and dollars for the Illinois and Wisconsin plants. A positive variance means Total Actual Hours are less than Total Standard Hours. Use the median wage for All Occupations in the food manufacturing industry from the BLS wage benchmark file to convert from hours to dollars.

Also give me the straight productivity index (Actual Hours / Standard Hours) for each plant.

Round hours to 2 decimal places, dollars to 2 decimal places, and the index to 2 decimal places.""",
        "ground_truth": truth["task9"],
        "rubric": rubric9,
    })
    
    # TASK 10
    tasks.append({
        "task_id": "task_10",
        "task_name": "Updated Productivity Loss with New Wages",
        "prompt": """The client sent us employee wage data (attached), so we need to update our assumptions. Find the average hourly salary across all employee roles in the attached wage file and use that to calculate the updated annual productivity loss for the entire company.

Note that survey responses represent one week of work. Report the annual productivity loss in thousands (000s) rounded to the nearest thousand. Also state the average hourly wage used.

Report your answer here.""",
        "ground_truth": truth["task10"],
        "rubric": [
            f"States the updated annual productivity loss is ${truth['task10']['annual_productivity_loss']:,}",
            f"States the average fully-loaded hourly wage is ${truth['task10']['avg_hourly_wage']}",
        ]
    })
    
    # TASK 11
    rubric11 = []
    for plant, data in truth["task11"]["plant_results"].items():
        rubric11.append(f"States that the unit sales for {plant} after deploying initiatives is {data['new_unit_sales']:,}")
        rubric11.append(f"States that the Revised Projected Sales for {plant} is ${data['new_projected_sales']:,}")
    tasks.append({
        "task_id": "task_11",
        "task_name": "Technology Investment Impact",
        "prompt": """Identify the top five technology investments from the Aptean report with the largest positive difference in percentage revenue growth between users and non-users. Include only investments that the report explicitly identifies as either top technology investments to date or top investments planned for 2024.

Next, assume that HarFeast will deploy all five of these top initiatives at every plant location. Apply the cumulative growth impact to each plant's current unit sales and calculate the revised projected sales revenue.

Round unit sales to the nearest whole number and revenue to the nearest dollar.""",
        "ground_truth": truth["task11"],
        "rubric": rubric11,
    })
    
    # TASK 12
    t12 = truth["task12"]
    tasks.append({
        "task_id": "task_12",
        "task_name": "Digital Adoption Willingness Analysis",
        "prompt": """To implement the required roadmap, we need to identify what roles and plants are most and least willing to go through a digital transformation.

Determine the plant with the highest and lowest average willingness to adopt digital tools. Within those plants, identify the roles with the highest and lowest willingness. For those specific role-plant combinations, determine the preferred training length, the count of employees preferring 1-2 days of training, and the total training cost (at $8/hour training rate).

Report your findings here.""",
        "ground_truth": truth["task12"],
        "rubric": [
            f"States that the plant with lowest willingness to adopt is {t12['lowest_willingness_plant']}",
            f"States that the plant with highest willingness to adopt is {t12['highest_willingness_plant']}",
            f"States the role with lowest willingness in {t12['lowest_willingness_plant']} is {t12['lowest_role_in_lowest_plant'][0]}",
            f"States the role with highest willingness in {t12['highest_willingness_plant']} is {t12['highest_role_in_highest_plant'][0]}",
        ]
    })
    
    # TASK 13
    rubric13 = [f"States that the new unplanned downtime ratio for {plant} is {pct}%" for plant, pct in truth["task13"].items()]
    tasks.append({
        "task_id": "task_13",
        "task_name": "Frito-Lay Downtime Reduction Application",
        "prompt": """Can you look at the Frito-Lay case study and apply their downtime reduction to HarFeast's numbers in the equipment data? I want to estimate what the improvement would look like for us (rounded to the nearest full percentage point).

Calculate the current unplanned downtime ratio (unplanned downtime hours / scheduled hours) for each plant, apply the reduction from the case study, and report the new ratios.

Output the information in a message here.""",
        "ground_truth": truth["task13"],
        "rubric": rubric13,
    })
    
    # TASK 14
    rubric14 = [f"States that the number of respondents who received training is {truth['task14']['trained_count']}"]
    for quality, pct in truth["task14"]["quality_pcts"].items():
        rubric14.append(f"States that percentage of respondents rated training as \"{quality}\" is {pct}%")
    tasks.append({
        "task_id": "task_14",
        "task_name": "Training Quality Assessment",
        "prompt": """Use the workforce survey responses to identify the number of respondents who received any kind of training on digital tools. Of those respondents, return the percentage of respondents for each training quality rating.

Reply back here to me.""",
        "ground_truth": truth["task14"],
        "rubric": rubric14,
    })
    
    return tasks


# =============================================================================
# FILE WRITERS
# =============================================================================

def write_csv(filepath, data, fieldnames=None):
    """Write a list of dicts to CSV."""
    if not data:
        return
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def write_text(filepath, content):
    """Write text content to a file."""
    with open(filepath, "w") as f:
        f.write(content)


# =============================================================================
# MAIN
# =============================================================================

def generate_world(seed=42, output_dir="./harfeast_world"):
    """Generate the complete HarFeast synthetic world."""
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "documents"), exist_ok=True)
    
    print("Generating data...")
    
    # Generate all datasets
    employees = generate_employee_survey(rng)
    equipment = generate_equipment_data(rng)
    quality_losses = generate_quality_losses(rng, equipment)
    plant_labor = generate_plant_labor(rng)
    bls_wages = generate_bls_wages()
    attached_wages = generate_attached_wages()
    oee_assumptions = generate_oee_assumptions()
    plant_sales = generate_plant_sales()
    aptean_data = generate_aptean_report()
    
    # Write CSV files
    write_csv(os.path.join(output_dir, "data", "employee_survey.csv"), employees)
    write_csv(os.path.join(output_dir, "data", "equipment_data.csv"), equipment)
    write_csv(os.path.join(output_dir, "data", "quality_losses.csv"), quality_losses)
    write_csv(os.path.join(output_dir, "data", "plant_labor.csv"), plant_labor)
    write_csv(os.path.join(output_dir, "data", "bls_wage_benchmark.csv"), bls_wages)
    write_csv(os.path.join(output_dir, "data", "attached_wage_data.csv"), attached_wages)
    write_csv(os.path.join(output_dir, "data", "oee_assumptions.csv"), oee_assumptions)
    write_csv(os.path.join(output_dir, "data", "plant_unit_sales.csv"), plant_sales)
    write_csv(os.path.join(output_dir, "data", "aptean_report_data.csv"), aptean_data)
    
    # Write text documents
    write_text(os.path.join(output_dir, "documents", "scrap_rate_report.txt"), generate_scrap_report())
    
    interviews = generate_interviews()
    for name, text in interviews.items():
        write_text(os.path.join(output_dir, "documents", f"interview_{name}.txt"), text)
    
    write_text(os.path.join(output_dir, "documents", "frito_lay_case_study.txt"), generate_frito_lay_case())
    write_text(os.path.join(output_dir, "documents", "aptean_report.txt"), generate_aptean_report_text(aptean_data))
    
    # Compute ground truth
    print("Computing ground truth...")
    truth = compute_ground_truth(
        employees, equipment, quality_losses, plant_labor,
        bls_wages, attached_wages, oee_assumptions, plant_sales, aptean_data
    )
    
    # Generate task prompts and rubrics
    print("Generating tasks...")
    tasks = generate_task_prompts(truth)
    
    # Write tasks and ground truth
    with open(os.path.join(output_dir, "tasks.json"), "w") as f:
        json.dump(tasks, f, indent=2, default=str)
    
    with open(os.path.join(output_dir, "ground_truth.json"), "w") as f:
        json.dump(truth, f, indent=2, default=str)
    
    # Print summary
    print(f"\nWorld generated in {output_dir}/")
    print(f"  Employees: {len(employees)}")
    print(f"  Equipment: {len(equipment)}")
    print(f"  Quality losses: {len(quality_losses)}")
    print(f"  Plant labor: {len(plant_labor)}")
    print(f"  Tasks: {len(tasks)}")
    print(f"\nGround truth summary:")
    for task in tasks:
        n_criteria = len(task["rubric"])
        print(f"  {task['task_id']} ({task['task_name']}): {n_criteria} criteria")
    
    # Print sample ground truth values for validation
    print(f"\nSample answers for validation:")
    print(f"  Task 1 - High-priority count: {truth['task1']['high_priority_count']}")
    print(f"  Task 6 - Avg inefficient hours: {truth['task6']['avg_by_plant']}")
    print(f"  Task 14 - Trained count: {truth['task14']['trained_count']}")
    print(f"  Task 13 - Downtime ratios: {truth['task13']}")
    
    return employees, equipment, truth, tasks


if __name__ == "__main__":
    import sys
    seed = 42
    output_dir = "./harfeast_world"
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--seed" and i + 2 < len(sys.argv):
            seed = int(sys.argv[i + 2])
        elif arg == "--output-dir" and i + 2 < len(sys.argv):
            output_dir = sys.argv[i + 2]
    
    generate_world(seed=seed, output_dir=output_dir)
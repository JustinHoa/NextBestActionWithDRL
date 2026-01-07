"""
Simulation with LP Batch Scheduling.

This simulation runs batch scheduling every 5 minutes for concurrent activities,
implementing true Linear Programming-based resource allocation.
"""

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import simpy

from agents.lp_batch_scheduler import LPBatchScheduler


# Activity definitions
ACTIVITY_NAMES = [
    "Registration", "Payment", "Measure Vital Signs", "General Medicine Examination",
    "Eye Examination", "ENT Examination", "Dental Examination", 
    "Gynecological Examination", "Breast Examination",
    "Blood Test", "Urine Test", "General Ultrasound", 
    "Cardiac Ultrasound", "Chest X-ray",
    "Conclusion"
]

CLUSTER_1 = ["Eye Examination", "ENT Examination", "Dental Examination", 
             "Gynecological Examination", "Breast Examination"]
CLUSTER_2 = ["DEXA Bone Density Scan", "Chest X-ray", "In-depth Eye Examination", "General Ultrasound", 
             "Urine Test", "ENT Endoscopy", "Electrocardiogram (ECG)", "Post-bronchodilator Spirometry", 
             "Cardiac Ultrasound", "Blood Test"]

# Processing times (mean in minutes)
PROCESSING_TIMES = {
    "Registration": 5.0,
    "Payment": 3.0,
    "Measure Vital Signs": 5.0,
    "General Medicine Examination": 10.0,
    "Eye Examination": 8.0,
    "ENT Examination": 8.0,
    "Dental Examination": 8.0,
    "Gynecological Examination": 10.0,
    "Breast Examination": 8.0,
    "DEXA Bone Density Scan": 15.0,
    "Chest X-ray": 8.0,
    "In-depth Eye Examination": 12.0,
    "General Ultrasound": 10.0,
    "Urine Test": 5.0,
    "ENT Endoscopy": 15.0,
    "Electrocardiogram (ECG)": 10.0,
    "Post-bronchodilator Spirometry": 20.0,
    "Cardiac Ultrasound": 12.0,
    "Blood Test": 5.0,
    "Conclusion": 5.0,
}


class HealthCheckCenter:
    """Healthcare center with resources for each activity."""
    
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.resources = {}
        self.queue_counts = defaultdict(int)
        self.finished_patient_count = 0
        
        # Create resources based on staff capacity
        capacities = {
            "Registration": 2,
            "Payment": 2,
            "Measure Vital Signs": 3,
            "General Medicine Examination": 3,
            "Eye Examination": 2,
            "ENT Examination": 2,
            "Dental Examination": 2,
            "Gynecological Examination": 1,
            "Breast Examination": 1,
            "DEXA Bone Density Scan": 1,
            "Chest X-ray": 2,
            "In-depth Eye Examination": 1,
            "General Ultrasound": 2,
            "Urine Test": 2,
            "ENT Endoscopy": 1,
            "Electrocardiogram (ECG)": 2,
            "Post-bronchodilator Spirometry": 1,
            "Cardiac Ultrasound": 1,
            "Blood Test": 3,
            "Conclusion": 2,
        }
        
        for activity, capacity in capacities.items():
            self.resources[activity] = simpy.Resource(env, capacity=capacity)
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get current queue length for all activities."""
        return {
            activity: len(resource.queue)
            for activity, resource in self.resources.items()
        }


class BatchCoordinator:
    """Coordinator that runs LP batch scheduling every 5 minutes."""
    
    def __init__(self, scheduler: LPBatchScheduler, batch_interval: float = 5.0):
        self.scheduler = scheduler
        self.batch_interval = batch_interval
        self.waiting_patients = {}  # patient_id -> patient_info
        self.assignments = {}  # patient_id -> assigned_activity
    
    def register_waiting_patient(
        self,
        patient_id: int,
        gender: str,
        marital: str,
        candidates: List[str],
        wait_start: float
    ):
        """Register a patient waiting for batch assignment."""
        self.waiting_patients[patient_id] = {
            'id': patient_id,
            'gender': gender,
            'marital': marital,
            'candidates': candidates,
            'wait_start': wait_start,
        }
    
    def get_assignment(self, patient_id: int) -> Optional[str]:
        """Get assigned activity for a patient (if available)."""
        return self.assignments.pop(patient_id, None)
    
    def remove_waiting_patient(self, patient_id: int):
        """Remove patient from waiting list (e.g., already assigned)."""
        self.waiting_patients.pop(patient_id, None)
    
    def run_batch_scheduling(self, center: HealthCheckCenter, current_time: float):
        """Run LP batch scheduling for all waiting patients."""
        if not self.waiting_patients:
            return
        
        waiting_list = list(self.waiting_patients.values())
        queue_status = center.get_queue_status()
        
        # Solve LP
        new_assignments = self.scheduler.solve_batch_assignment(
            waiting_list, queue_status, current_time
        )
        
        # Store assignments
        self.assignments.update(new_assignments)
        
        # Remove assigned patients from waiting list
        for patient_id in new_assignments.keys():
            self.waiting_patients.pop(patient_id, None)


class Patient:
    """Patient in healthcare system."""
    
    def __init__(
        self,
        env: simpy.Environment,
        center: HealthCheckCenter,
        patient_id: int,
        gender: str,
        marital: str,
        event_logs: List,
    ):
        self.env = env
        self.center = center
        self.id = patient_id
        self.gender = gender
        self.marital = marital
        self.event_logs = event_logs
        self.start_time = 0.0
        self.end_time = 0.0
    
    def do_activity(self, activity: str):
        """Perform an activity with resource request."""
        resource = self.center.resources[activity]
        
        # Log start
        self.event_logs.append({
            'PatientID': self.id,
            'Activity': activity,
            'Lifecycle': 'START',
            'Timestamp': self.env.now,
        })
        
        with resource.request() as req:
            yield req
            
            # Processing time
            proc_time = np.random.exponential(PROCESSING_TIMES.get(activity, 5.0))
            yield self.env.timeout(proc_time)
        
        # Log complete
        self.event_logs.append({
            'PatientID': self.id,
            'Activity': activity,
            'Lifecycle': 'COMPLETE',
            'Timestamp': self.env.now,
        })
    
    def go_process(self, coordinator: BatchCoordinator):
        """Patient journey with batch scheduling for concurrent activities."""
        self.start_time = self.env.now
        
        # Filter candidates based on gender/marital
        c1_candidates = [
            a for a in CLUSTER_1
            if not (
                (a == "Gynecological Examination" and (self.gender == "Male" or self.marital == "Single"))
                or (a == "Breast Examination" and self.gender == "Male")
            )
        ]
        c2_candidates = list(CLUSTER_2)
        
        # 1. Sequential activities (no batch scheduling needed)
        for activity in ["Registration", "Payment", "Measure Vital Signs", "General Medicine Examination"]:
            yield self.env.process(self.do_activity(activity))
        
        # 2. CLUSTER 1 - Batch scheduling
        while c1_candidates:
            # Register for batch scheduling
            coordinator.register_waiting_patient(
                self.id, self.gender, self.marital, c1_candidates, self.env.now
            )
            
            # Wait for assignment (check every 0.1 min)
            while True:
                assignment = coordinator.get_assignment(self.id)
                if assignment:
                    break
                yield self.env.timeout(0.1)
            
            # Perform assigned activity
            c1_candidates.remove(assignment)
            yield self.env.process(self.do_activity(assignment))
        
        # 3. CLUSTER 2 - Batch scheduling
        while c2_candidates:
            # Register for batch scheduling
            coordinator.register_waiting_patient(
                self.id, self.gender, self.marital, c2_candidates, self.env.now
            )
            
            # Wait for assignment
            while True:
                assignment = coordinator.get_assignment(self.id)
                if assignment:
                    break
                yield self.env.timeout(0.1)
            
            # Perform assigned activity
            c2_candidates.remove(assignment)
            yield self.env.process(self.do_activity(assignment))
        
        # 4. Final activity
        yield self.env.process(self.do_activity("Conclusion"))
        
        self.end_time = self.env.now
        self.center.finished_patient_count += 1


def batch_scheduler_process(env: simpy.Environment, coordinator: BatchCoordinator, center: HealthCheckCenter):
    """Process that runs batch scheduling every 5 minutes."""
    while True:
        # Run scheduling first, then wait
        coordinator.run_batch_scheduling(center, env.now)
        yield env.timeout(coordinator.batch_interval)


def run_lp_batch_simulation(
    num_patients: int = 200,
    scheduler: Optional[LPBatchScheduler] = None,
    version_output: str = "lp_batch",
    seed: Optional[int] = None,
    model_name: str = "LPBatch",
    gen_id: int = 0,
    batch_interval: float = 5.0,
) -> float:
    """
    Run simulation with LP batch scheduling.
    
    Args:
        num_patients: Number of patients to simulate
        scheduler: LP batch scheduler (if None, creates default)
        version_output: Output version tag
        seed: Random seed
        model_name: Model name for logging
        gen_id: Generation ID
        batch_interval: Batch scheduling interval in minutes
    
    Returns:
        Average patient time in minutes
    """
    # Set seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create scheduler if not provided
    if scheduler is None:
        scheduler = LPBatchScheduler(state_size=45, action_size=21)
    
    # Setup simulation
    env = simpy.Environment()
    center = HealthCheckCenter(env)
    coordinator = BatchCoordinator(scheduler, batch_interval)
    
    event_logs = []
    patients_list = []
    
    # Patient generator
    def patient_gen():
        for i in range(num_patients):
            gender = random.choice(["Male", "Female"])
            marital = random.choice(["Single", "Married"])
            patient = Patient(env, center, i, gender, marital, event_logs)
            patients_list.append(patient)
            env.process(patient.go_process(coordinator))
            yield env.timeout(random.expovariate(1.0 / 2.0))
    
    # Start processes
    env.process(patient_gen())
    env.process(batch_scheduler_process(env, coordinator, center))
    
    # Run simulation
    env.run()
    
    # Calculate average time
    total_time = sum(p.end_time - p.start_time for p in patients_list)
    avg_time = total_time / num_patients if num_patients > 0 else 0.0
    
    # Save event log
    df_events = pd.DataFrame(event_logs)
    eval_dir = os.path.join("data", "evaluate")
    os.makedirs(eval_dir, exist_ok=True)
    
    event_path = os.path.join(
        eval_dir,
        f"event_log_{num_patients}_{model_name}_gen_{gen_id}_checkpoint_0.csv"
    )
    df_events.to_csv(event_path, index=False)
    print(f"✅ Saved EventLog: {event_path}")
    print(f"✅ LP Batch Simulation finished: Avg Time = {avg_time:.2f} mins")
    
    return avg_time

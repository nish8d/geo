"""
Donor-Driven Blood Request Matching System
Modified to show nearby requests to donors instead of finding donors for requests
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, date
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BloodType(Enum):
    """Blood type enumeration"""
    O_NEGATIVE = "O-"
    O_POSITIVE = "O+"
    A_NEGATIVE = "A-"
    A_POSITIVE = "A+"
    B_NEGATIVE = "B-"
    B_POSITIVE = "B+"
    AB_NEGATIVE = "AB-"
    AB_POSITIVE = "AB+"

class UrgencyLevel(Enum):
    """Blood request urgency levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Donor:
    """Data class representing a blood donor"""
    id: str
    full_name: str
    email: str
    phone: str
    blood_group: str
    latitude: float
    longitude: float
    location_text: str
    formatted_address: str
    available_date: str
    available_time: str
    last_donation_date: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None

@dataclass
class BloodRequest:
    """Data class representing a blood request"""
    id: str
    patient_name: str
    contact_email: str
    contact_phone: str
    blood_group: str
    hospital_name: str
    latitude: float
    longitude: float
    hospital_address: str
    urgency: str
    units_needed: int
    required_date: str
    required_time: Optional[str] = None
    created_at: Optional[datetime] = None
    is_active: bool = True  # Added to manage active/fulfilled requests

@dataclass
class RequestMatch:
    """Data class for request-donor match results (flipped perspective)"""
    request: BloodRequest
    donor: Donor
    distance_km: float
    compatibility_score: float
    is_blood_compatible: bool
    is_date_compatible: bool
    urgency_priority: int
    overall_score: float
    can_donate_safely: bool  # New field for donor safety check

class DonorDrivenMatchingEngine:
    """
    Engine for showing compatible blood requests to donors based on their location and blood type
    """
    
    def __init__(self):
        """Initialize the matching engine"""
        self.donor_compatibility_matrix = self._build_donor_compatibility_matrix()
        self.urgency_weights = {
            UrgencyLevel.CRITICAL: 4.0,
            UrgencyLevel.HIGH: 3.0,
            UrgencyLevel.NORMAL: 2.0,
            UrgencyLevel.LOW: 1.0
        }
    
    def _build_donor_compatibility_matrix(self) -> Dict[str, List[str]]:
        """
        Build compatibility matrix from donor perspective
        Returns dict where keys are donor blood types and values are patient types they can help
        """
        return {
            "O-": ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],  # Universal donor
            "O+": ["O+", "A+", "B+", "AB+"],
            "A-": ["A-", "A+", "AB-", "AB+"],
            "A+": ["A+", "AB+"],
            "B-": ["B-", "B+", "AB-", "AB+"],
            "B+": ["B+", "AB+"],
            "AB-": ["AB-", "AB+"],
            "AB+": ["AB+"]  # Can only donate to AB+
        }
    
    def validate_coordinates(self, latitude: float, longitude: float) -> bool:
        """Validate coordinates are within valid ranges"""
        return (-90 <= latitude <= 90) and (-180 <= longitude <= 180)
    
    def validate_blood_group(self, blood_group: str) -> bool:
        """Validate blood group format"""
        valid_groups = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
        return blood_group in valid_groups
    
    def validate_date(self, date_string: str) -> bool:
        """Validate date format"""
        try:
            datetime.strptime(date_string, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates using Haversine formula"""
        if not all([self.validate_coordinates(lat1, lon1), self.validate_coordinates(lat2, lon2)]):
            raise ValueError("Invalid coordinates provided")
        
        R = 6371  # Earth's radius in kilometers
        
        # Convert degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Calculate differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def can_donor_help_patient(self, donor_blood: str, patient_blood: str) -> bool:
        """
        Check if donor can safely donate to patient
        """
        if not (self.validate_blood_group(donor_blood) and self.validate_blood_group(patient_blood)):
            logger.warning(f"Invalid blood groups: donor={donor_blood}, patient={patient_blood}")
            return False
        
        return patient_blood in self.donor_compatibility_matrix.get(donor_blood, [])
    
    def is_donor_eligible_to_donate(self, donor: Donor, required_date: str) -> Tuple[bool, str]:
        """
        Check if donor is eligible to donate based on last donation date and availability
        """
        # Check if donor is available on required date
        try:
            if not (self.validate_date(donor.available_date) and self.validate_date(required_date)):
                return False, "Invalid date format"
            
            donor_date = datetime.strptime(donor.available_date, "%Y-%m-%d").date()
            req_date = datetime.strptime(required_date, "%Y-%m-%d").date()
            
            if donor_date > req_date:
                return False, "Donor not available by required date"
            
        except ValueError:
            return False, "Date parsing error"
        
        # Check donation interval (56 days minimum between donations)
        if donor.last_donation_date:
            try:
                last_donation = datetime.strptime(donor.last_donation_date, "%Y-%m-%d").date()
                days_since_donation = (date.today() - last_donation).days
                
                if days_since_donation < 56:
                    days_remaining = 56 - days_since_donation
                    return False, f"Must wait {days_remaining} more days since last donation"
                    
            except ValueError:
                pass  # If date is invalid, assume eligible
        
        return True, "Eligible to donate"
    
    def calculate_request_appeal_score(self, donor: Donor, request: BloodRequest) -> float:
        """
        Calculate how appealing a request should be to a donor
        """
        score = 0.0
        
        try:
            # Blood compatibility (40 points max)
            if self.can_donor_help_patient(donor.blood_group, request.blood_group):
                score += 40.0
                # Bonus for being able to help rare blood types
                if request.blood_group in ["AB-", "AB+", "O-"]:
                    score += 5.0
                # Bonus for exact blood type match
                if donor.blood_group == request.blood_group:
                    score += 5.0
            
            # Urgency factor (25 points max)
            try:
                urgency_level = UrgencyLevel(request.urgency.lower())
                urgency_multiplier = self.urgency_weights[urgency_level]
                score += urgency_multiplier * 6.25  # Scale to 25 points max
            except (ValueError, KeyError):
                score += 12.5  # Default normal urgency
            
            # Date compatibility (20 points max)
            eligible, _ = self.is_donor_eligible_to_donate(donor, request.required_date)
            if eligible:
                score += 20.0
                
                # Bonus for immediate availability
                try:
                    donor_date = datetime.strptime(donor.available_date, "%Y-%m-%d").date()
                    req_date = datetime.strptime(request.required_date, "%Y-%m-%d").date()
                    days_diff = (req_date - donor_date).days
                    
                    if days_diff <= 1:
                        score += 10.0
                    elif days_diff <= 3:
                        score += 5.0
                except ValueError:
                    pass
            
            # Units needed factor (15 points max)
            if request.units_needed == 1:
                score += 15.0  # Standard single unit
            elif request.units_needed == 2:
                score += 10.0  # Common double unit
            else:
                score += 5.0   # Multiple units (may require multiple donors)
            
        except Exception as e:
            logger.error(f"Error calculating appeal score: {e}")
            return 0.0
        
        return min(score, 100.0)  # Cap at 100
    
    def calculate_urgency_priority(self, urgency: str) -> int:
        """Calculate urgency priority score"""
        try:
            urgency_enum = UrgencyLevel(urgency.lower())
            return int(self.urgency_weights[urgency_enum])
        except (ValueError, KeyError):
            logger.warning(f"Invalid urgency level: {urgency}, defaulting to normal")
            return 2
    
    def calculate_overall_appeal_score(self, distance: float, appeal_score: float, 
                                     urgency_priority: int, max_distance: float = 50.0) -> float:
        """
        Calculate overall appeal score for a request from donor's perspective
        """
        try:
            # Distance score (closer requests are more appealing)
            if distance <= max_distance:
                distance_score = (max_distance - distance) / max_distance * 30.0
            else:
                distance_score = 0.0
            
            # Appeal score (0-50 points)
            appeal_points = appeal_score * 0.5
            
            # Urgency multiplier (critical cases get priority)
            urgency_multiplier = 1.0 + (urgency_priority - 1) * 0.2
            
            # Base score
            base_score = distance_score + appeal_points
            
            # Apply urgency multiplier
            final_score = min(base_score * urgency_multiplier, 100.0)
            
            return final_score
        except Exception as e:
            logger.error(f"Error calculating overall appeal score: {e}")
            return 0.0
    
    def find_nearby_requests_for_donor(self, donor: Donor, requests: List[BloodRequest], 
                                     max_distance: float = 30.0, max_results: int = 10) -> List[RequestMatch]:
        """
        Find nearby compatible blood requests for a specific donor
        
        Args:
            donor: The donor looking for requests to help with
            requests: List of active blood requests
            max_distance: Maximum search distance in kilometers  
            max_results: Maximum number of results to return
            
        Returns:
            List[RequestMatch]: Sorted list of compatible requests
        """
        if not donor or not requests:
            logger.warning("Missing donor or requests data")
            return []
        
        if not donor.is_active:
            logger.warning(f"Donor {donor.id} is not active")
            return []
        
        matches = []
        
        try:
            for request in requests:
                # Skip inactive/fulfilled requests
                if not request.is_active:
                    continue
                
                # Validate request data
                if not (self.validate_coordinates(request.latitude, request.longitude) and
                        self.validate_blood_group(request.blood_group) and
                        self.validate_date(request.required_date)):
                    logger.warning(f"Invalid request data for request {request.id}")
                    continue
                
                # Calculate distance
                try:
                    distance = self.calculate_distance(
                        donor.latitude, donor.longitude,
                        request.latitude, request.longitude
                    )
                except ValueError as e:
                    logger.error(f"Distance calculation failed for request {request.id}: {e}")
                    continue
                
                # Skip requests beyond max distance
                if distance > max_distance:
                    continue
                
                # Check if donor can help this patient
                can_help = self.can_donor_help_patient(donor.blood_group, request.blood_group)
                if not can_help:
                    continue  # Skip incompatible requests
                
                # Check if donor is eligible to donate
                eligible, eligibility_reason = self.is_donor_eligible_to_donate(donor, request.required_date)
                
                # Calculate appeal score
                appeal_score = self.calculate_request_appeal_score(donor, request)
                
                # Calculate urgency priority
                urgency_priority = self.calculate_urgency_priority(request.urgency)
                
                # Calculate overall appeal score
                overall_score = self.calculate_overall_appeal_score(
                    distance, appeal_score, urgency_priority, max_distance
                )
                
                # Create match result
                match = RequestMatch(
                    request=request,
                    donor=donor,
                    distance_km=round(distance, 2),
                    compatibility_score=round(appeal_score, 2),
                    is_blood_compatible=can_help,
                    is_date_compatible=eligible,
                    urgency_priority=urgency_priority,
                    overall_score=round(overall_score, 2),
                    can_donate_safely=eligible
                )
                
                matches.append(match)
        
        except Exception as e:
            logger.error(f"Error in find_nearby_requests_for_donor: {e}")
            return []
        
        # Sort by overall score (highest first), then by urgency (critical first), then by distance (nearest first)
        matches.sort(key=lambda x: (-x.overall_score, -x.urgency_priority, x.distance_km))
        
        return matches[:max_results]
    
    def get_donor_dashboard_summary(self, donor: Donor, requests: List[BloodRequest]) -> Dict[str, Any]:
        """
        Get summary information for donor dashboard
        """
        try:
            # Find all nearby requests (within larger radius for summary)
            nearby_matches = self.find_nearby_requests_for_donor(donor, requests, max_distance=50.0, max_results=50)
            
            # Categorize by urgency
            critical_requests = [m for m in nearby_matches if m.request.urgency == 'critical']
            high_requests = [m for m in nearby_matches if m.request.urgency == 'high']
            normal_requests = [m for m in nearby_matches if m.request.urgency == 'normal']
            
            # Check eligibility
            eligible, eligibility_reason = self.is_donor_eligible_to_donate(donor, date.today().strftime("%Y-%m-%d"))
            
            return {
                "donor_id": donor.id,
                "donor_name": donor.full_name,
                "is_eligible": eligible,
                "eligibility_reason": eligibility_reason,
                "total_nearby_requests": len(nearby_matches),
                "critical_requests": len(critical_requests),
                "high_priority_requests": len(high_requests),
                "normal_requests": len(normal_requests),
                "closest_request_distance": nearby_matches[0].distance_km if nearby_matches else None,
                "blood_types_you_can_help": self.donor_compatibility_matrix.get(donor.blood_group, []),
                "next_available_date": donor.available_date
            }
        except Exception as e:
            logger.error(f"Error getting donor dashboard summary: {e}")
            return {"error": str(e)}

def example_usage():
    """Demonstrate the donor-driven matching system"""
    
    # Create sample blood requests (these would come from hospitals)
    requests = [
        BloodRequest(
            id="R001", patient_name="Emergency Patient A", contact_email="hospital1@kem.com",
            contact_phone="+912234567890", blood_group="A+", hospital_name="KEM Hospital",
            latitude=18.9894, longitude=72.8318, hospital_address="KEM Hospital, Parel, Mumbai",
            urgency="critical", units_needed=2, required_date="2024-03-16", is_active=True
        ),
        BloodRequest(
            id="R002", patient_name="Surgery Patient B", contact_email="hospital2@lilavati.com",
            contact_phone="+912234567891", blood_group="O+", hospital_name="Lilavati Hospital",
            latitude=19.0596, longitude=72.8295, hospital_address="Lilavati Hospital, Bandra, Mumbai",
            urgency="high", units_needed=1, required_date="2024-03-17", is_active=True
        ),
        BloodRequest(
            id="R003", patient_name="Routine Patient C", contact_email="hospital3@hinduja.com",
            contact_phone="+912234567892", blood_group="AB+", hospital_name="Hinduja Hospital",
            latitude=19.0596, longitude=72.8295, hospital_address="Hinduja Hospital, Mahim, Mumbai",
            urgency="normal", units_needed=1, required_date="2024-03-20", is_active=True
        )
    ]
    
    # Create a sample donor
    donor = Donor(
        id="D001", full_name="John Doe", email="john.doe@email.com", phone="+912234567893",
        blood_group="O+", latitude=19.0760, longitude=72.8777,
        location_text="Mumbai, Maharashtra", formatted_address="Mumbai, Maharashtra, India",
        available_date="2024-03-15", available_time="09:00-17:00", is_active=True
    )
    
    # Initialize matching engine
    matcher = DonorDrivenMatchingEngine()
    
    print("=== Donor-Driven Blood Request Matching Demo ===\n")
    
    # Get donor dashboard summary
    print(f"Dashboard Summary for {donor.full_name} ({donor.blood_group}):")
    summary = matcher.get_donor_dashboard_summary(donor, requests)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print()
    
    # Find nearby requests for the donor
    print(f"Nearby Blood Requests for {donor.full_name}:")
    matches = matcher.find_nearby_requests_for_donor(donor, requests, max_distance=30.0, max_results=10)
    
    if not matches:
        print("No compatible requests found nearby.")
    else:
        print(f"Found {len(matches)} requests you can help with:\n")
        
        for i, match in enumerate(matches, 1):
            request = match.request
            print(f"{i}. Patient: {request.patient_name}")
            print(f"   Blood Group Needed: {request.blood_group}")
            print(f"   Hospital: {request.hospital_name}")
            print(f"   Distance: {match.distance_km} km from you")
            print(f"   Urgency: {request.urgency.upper()}")
            print(f"   Units Needed: {request.units_needed}")
            print(f"   Required Date: {request.required_date}")
            print(f"   Appeal Score: {match.overall_score}/100")
            print(f"   Can Donate Safely: {'Yes' if match.can_donate_safely else 'No'}")
            print(f"   Contact: {request.contact_email}, {request.contact_phone}")
            print(f"   Hospital Address: {request.hospital_address}")
            print()

if __name__ == "__main__":
    example_usage()

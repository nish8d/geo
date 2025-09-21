"""
Donor Matching Algorithm - Find Nearest Compatible Donors
Calculates nearest donors to blood requests based on multiple criteria
Enhanced with better validation and error handling
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert donor to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Donor':
        """Create donor from dictionary"""
        return cls(**data)


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert blood request to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BloodRequest':
        """Create blood request from dictionary"""
        return cls(**data)


@dataclass
class MatchResult:
    """Data class for donor-request match results"""
    donor: Donor
    request: BloodRequest
    distance_km: float
    compatibility_score: float
    is_blood_compatible: bool
    is_date_compatible: bool
    urgency_priority: int
    overall_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert match result to dictionary"""
        return {
            'donor': self.donor.to_dict(),
            'request': self.request.to_dict(),
            'distance_km': self.distance_km,
            'compatibility_score': self.compatibility_score,
            'is_blood_compatible': self.is_blood_compatible,
            'is_date_compatible': self.is_date_compatible,
            'urgency_priority': self.urgency_priority,
            'overall_score': self.overall_score
        }


class DonorMatchingEngine:
    """
    Engine for matching blood donors with requests based on proximity and compatibility
    """
    
    def __init__(self):
        """Initialize the matching engine"""
        self.blood_compatibility_matrix = self._build_compatibility_matrix()
        self.urgency_weights = {
            UrgencyLevel.CRITICAL: 4.0,
            UrgencyLevel.HIGH: 3.0,
            UrgencyLevel.NORMAL: 2.0,
            UrgencyLevel.LOW: 1.0
        }
    
    def _build_compatibility_matrix(self) -> Dict[str, List[str]]:
        """
        Build blood type compatibility matrix
        Returns dict where keys are donor blood types and values are compatible recipient types
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
        """
        Validate coordinates are within valid ranges
        
        Args:
            latitude: Latitude value
            longitude: Longitude value
            
        Returns:
            bool: True if coordinates are valid
        """
        return (-90 <= latitude <= 90) and (-180 <= longitude <= 180)
    
    def validate_blood_group(self, blood_group: str) -> bool:
        """
        Validate blood group format
        
        Args:
            blood_group: Blood group string
            
        Returns:
            bool: True if blood group is valid
        """
        valid_groups = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
        return blood_group in valid_groups
    
    def validate_date(self, date_string: str) -> bool:
        """
        Validate date format
        
        Args:
            date_string: Date in YYYY-MM-DD format
            
        Returns:
            bool: True if date format is valid
        """
        try:
            datetime.strptime(date_string, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates using Haversine formula
        
        Args:
            lat1, lon1: Latitude and longitude of first point
            lat2, lon2: Latitude and longitude of second point
            
        Returns:
            float: Distance in kilometers
        """
        # Validate coordinates
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
    
    def is_blood_compatible(self, donor_blood: str, patient_blood: str) -> bool:
        """
        Check if donor blood type is compatible with patient blood type
        
        Args:
            donor_blood: Donor's blood type
            patient_blood: Patient's blood type
            
        Returns:
            bool: True if compatible, False otherwise
        """
        if not (self.validate_blood_group(donor_blood) and self.validate_blood_group(patient_blood)):
            logger.warning(f"Invalid blood groups: donor={donor_blood}, patient={patient_blood}")
            return False
        
        return patient_blood in self.blood_compatibility_matrix.get(donor_blood, [])
    
    def is_date_compatible(self, donor_available_date: str, required_date: str) -> bool:
        """
        Check if donor is available on or before required date
        
        Args:
            donor_available_date: Donor's available date (YYYY-MM-DD)
            required_date: Required donation date (YYYY-MM-DD)
            
        Returns:
            bool: True if donor is available, False otherwise
        """
        try:
            if not (self.validate_date(donor_available_date) and self.validate_date(required_date)):
                return False
            
            donor_date = datetime.strptime(donor_available_date, "%Y-%m-%d").date()
            req_date = datetime.strptime(required_date, "%Y-%m-%d").date()
            return donor_date <= req_date
        except ValueError as e:
            logger.error(f"Date validation error: {e}")
            return False
    
    def calculate_compatibility_score(self, donor: Donor, request: BloodRequest) -> float:
        """
        Calculate compatibility score based on multiple factors
        
        Args:
            donor: Donor object
            request: BloodRequest object
            
        Returns:
            float: Compatibility score (0-100)
        """
        score = 0.0
        
        try:
            # Blood compatibility (40 points max)
            if self.is_blood_compatible(donor.blood_group, request.blood_group):
                score += 40.0
                # Bonus for exact blood type match
                if donor.blood_group == request.blood_group:
                    score += 5.0
            
            # Date compatibility (25 points max)
            if self.is_date_compatible(donor.available_date, request.required_date):
                score += 25.0
                
                # Calculate days difference for bonus scoring
                try:
                    donor_date = datetime.strptime(donor.available_date, "%Y-%m-%d").date()
                    req_date = datetime.strptime(request.required_date, "%Y-%m-%d").date()
                    days_diff = (req_date - donor_date).days
                    
                    # Bonus for immediate availability
                    if days_diff <= 1:
                        score += 10.0
                    elif days_diff <= 3:
                        score += 5.0
                except ValueError:
                    pass
            
            # Active status (15 points max)
            if donor.is_active:
                score += 15.0
            
            # Recent donation check (10 points max)
            if donor.last_donation_date:
                try:
                    last_donation = datetime.strptime(donor.last_donation_date, "%Y-%m-%d").date()
                    days_since_donation = (date.today() - last_donation).days
                    
                    # Donors should wait at least 56 days between donations
                    if days_since_donation >= 56:
                        score += 10.0
                    elif days_since_donation >= 42:  # Partial eligibility
                        score += 5.0
                except ValueError:
                    score += 10.0  # No previous donation record
            else:
                score += 10.0  # First-time donor
            
        except Exception as e:
            logger.error(f"Error calculating compatibility score: {e}")
            return 0.0
        
        return min(score, 100.0)  # Cap at 100
    
    def calculate_urgency_priority(self, urgency: str) -> int:
        """
        Calculate urgency priority score
        
        Args:
            urgency: Urgency level string
            
        Returns:
            int: Priority score (1-4)
        """
        try:
            urgency_enum = UrgencyLevel(urgency.lower())
            return int(self.urgency_weights[urgency_enum])
        except (ValueError, KeyError):
            logger.warning(f"Invalid urgency level: {urgency}, defaulting to normal")
            return 2  # Default to normal priority
    
    def calculate_overall_score(self, distance: float, compatibility_score: float, 
                              urgency_priority: int, max_distance: float = 100.0) -> float:
        """
        Calculate overall matching score combining distance, compatibility, and urgency
        
        Args:
            distance: Distance in kilometers
            compatibility_score: Compatibility score (0-100)
            urgency_priority: Urgency priority (1-4)
            max_distance: Maximum distance for scoring
            
        Returns:
            float: Overall score (0-100)
        """
        try:
            # Distance score (closer is better)
            if distance <= max_distance:
                distance_score = (max_distance - distance) / max_distance * 40.0
            else:
                distance_score = 0.0
            
            # Compatibility score (0-40 points)
            compatibility_points = compatibility_score * 0.4
            
            # Urgency multiplier (1.0 to 2.0)
            urgency_multiplier = 1.0 + (urgency_priority - 1) * 0.25
            
            # Base score
            base_score = distance_score + compatibility_points
            
            # Apply urgency multiplier
            final_score = min(base_score * urgency_multiplier, 100.0)
            
            return final_score
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.0
    
    def find_nearest_donors(self, request: BloodRequest, donors: List[Donor], 
                           max_distance: float = 50.0, max_results: int = 10) -> List[MatchResult]:
        """
        Find nearest compatible donors for a blood request
        
        Args:
            request: BloodRequest object
            donors: List of available donors
            max_distance: Maximum search distance in kilometers
            max_results: Maximum number of results to return
            
        Returns:
            List[MatchResult]: Sorted list of matching donors
        """
        if not request or not donors:
            logger.warning("Missing request or donors data")
            return []
        
        matches = []
        
        try:
            for donor in donors:
                # Skip inactive donors
                if not donor.is_active:
                    continue
                
                # Validate donor data
                if not (self.validate_coordinates(donor.latitude, donor.longitude) and
                        self.validate_blood_group(donor.blood_group) and
                        self.validate_date(donor.available_date)):
                    logger.warning(f"Invalid donor data for donor {donor.id}")
                    continue
                
                # Calculate distance
                try:
                    distance = self.calculate_distance(
                        request.latitude, request.longitude,
                        donor.latitude, donor.longitude
                    )
                except ValueError as e:
                    logger.error(f"Distance calculation failed for donor {donor.id}: {e}")
                    continue
                
                # Skip donors beyond max distance
                if distance > max_distance:
                    continue
                
                # Calculate compatibility score
                compatibility_score = self.calculate_compatibility_score(donor, request)
                
                # Check blood compatibility
                blood_compatible = self.is_blood_compatible(donor.blood_group, request.blood_group)
                
                # Check date compatibility
                date_compatible = self.is_date_compatible(donor.available_date, request.required_date)
                
                # Skip if blood type is not compatible
                if not blood_compatible:
                    continue
                
                # Calculate urgency priority
                urgency_priority = self.calculate_urgency_priority(request.urgency)
                
                # Calculate overall score
                overall_score = self.calculate_overall_score(
                    distance, compatibility_score, urgency_priority, max_distance
                )
                
                # Create match result
                match = MatchResult(
                    donor=donor,
                    request=request,
                    distance_km=round(distance, 2),
                    compatibility_score=round(compatibility_score, 2),
                    is_blood_compatible=blood_compatible,
                    is_date_compatible=date_compatible,
                    urgency_priority=urgency_priority,
                    overall_score=round(overall_score, 2)
                )
                
                matches.append(match)
        
        except Exception as e:
            logger.error(f"Error in find_nearest_donors: {e}")
            return []
        
        # Sort by overall score (highest first), then by distance (nearest first)
        matches.sort(key=lambda x: (-x.overall_score, x.distance_km))
        
        return matches[:max_results]
    
    def find_donors_by_blood_group(self, blood_group: str, donors: List[Donor]) -> List[Donor]:
        """
        Find all donors with compatible blood types for a specific blood group
        
        Args:
            blood_group: Required blood group
            donors: List of all donors
            
        Returns:
            List[Donor]: Compatible donors
        """
        if not self.validate_blood_group(blood_group):
            logger.error(f"Invalid blood group: {blood_group}")
            return []
        
        compatible_donors = []
        
        for donor in donors:
            if (donor.is_active and 
                self.validate_blood_group(donor.blood_group) and
                self.is_blood_compatible(donor.blood_group, blood_group)):
                compatible_donors.append(donor)
        
        return compatible_donors
    
    def find_donors_in_radius(self, center_lat: float, center_lon: float, 
                             donors: List[Donor], radius_km: float = 25.0) -> List[Tuple[Donor, float]]:
        """
        Find all donors within a specific radius from a center point
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            donors: List of donors
            radius_km: Search radius in kilometers
            
        Returns:
            List[Tuple[Donor, float]]: List of (donor, distance) tuples sorted by distance
        """
        if not self.validate_coordinates(center_lat, center_lon):
            logger.error("Invalid center coordinates")
            return []
        
        donors_in_radius = []
        
        for donor in donors:
            if not (donor.is_active and self.validate_coordinates(donor.latitude, donor.longitude)):
                continue
            
            try:
                distance = self.calculate_distance(center_lat, center_lon, donor.latitude, donor.longitude)
                if distance <= radius_km:
                    donors_in_radius.append((donor, round(distance, 2)))
            except ValueError:
                continue
        
        # Sort by distance
        donors_in_radius.sort(key=lambda x: x[1])
        
        return donors_in_radius
    
    def get_matching_statistics(self, matches: List[MatchResult]) -> Dict[str, Any]:
        """
        Get statistics about matching results
        
        Args:
            matches: List of match results
            
        Returns:
            dict: Statistics dictionary
        """
        if not matches:
            return {"total_matches": 0}
        
        try:
            distances = [match.distance_km for match in matches]
            scores = [match.overall_score for match in matches]
            blood_groups = [match.donor.blood_group for match in matches]
            
            return {
                "total_matches": len(matches),
                "average_distance": round(sum(distances) / len(distances), 2),
                "min_distance": min(distances),
                "max_distance": max(distances),
                "average_score": round(sum(scores) / len(scores), 2),
                "blood_group_distribution": {bg: blood_groups.count(bg) for bg in set(blood_groups)},
                "date_compatible_count": sum(1 for match in matches if match.is_date_compatible)
            }
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {"total_matches": 0, "error": str(e)}


def example_usage():
    """Demonstrate the donor matching algorithm"""
    
    # Create sample donors
    donors = [
        Donor(
            id="D001", full_name="John Doe", email="john@email.com", phone="+1234567890",
            blood_group="O+", latitude=19.0760, longitude=72.8777,
            location_text="Mumbai, Maharashtra", formatted_address="Mumbai, Maharashtra, India",
            available_date="2024-03-15", available_time="09:00-17:00", is_active=True
        ),
        Donor(
            id="D002", full_name="Jane Smith", email="jane@email.com", phone="+1234567891",
            blood_group="A+", latitude=19.1136, longitude=72.8697,
            location_text="Andheri, Mumbai", formatted_address="Andheri West, Mumbai, Maharashtra, India",
            available_date="2024-03-16", available_time="10:00-16:00", is_active=True
        ),
        Donor(
            id="D003", full_name="Mike Johnson", email="mike@email.com", phone="+1234567892",
            blood_group="O-", latitude=19.0970, longitude=72.9036,
            location_text="Powai, Mumbai", formatted_address="Powai, Mumbai, Maharashtra, India",
            available_date="2024-03-14", available_time="08:00-18:00", is_active=True,
            last_donation_date="2023-12-01"
        ),
        Donor(
            id="D004", full_name="Sarah Wilson", email="sarah@email.com", phone="+1234567893",
            blood_group="B+", latitude=18.9220, longitude=72.8347,
            location_text="Worli, Mumbai", formatted_address="Worli, Mumbai, Maharashtra, India",
            available_date="2024-03-20", available_time="11:00-15:00", is_active=True
        )
    ]
    
    # Create sample blood request
    request = BloodRequest(
        id="R001", patient_name="Emergency Patient", contact_email="hospital@kem.com",
        contact_phone="+1234567894", blood_group="A+", hospital_name="KEM Hospital",
        latitude=18.9894, longitude=72.8318, hospital_address="KEM Hospital, Parel, Mumbai",
        urgency="high", units_needed=2, required_date="2024-03-16"
    )
    
    # Initialize matching engine
    matcher = DonorMatchingEngine()
    
    print("=== Donor Matching Algorithm Demo ===\n")
    
    # Find nearest donors
    print(f"Finding donors for patient: {request.patient_name}")
    print(f"Blood group needed: {request.blood_group}")
    print(f"Hospital: {request.hospital_name}")
    print(f"Urgency: {request.urgency}")
    print(f"Required date: {request.required_date}")
    print()
    
    matches = matcher.find_nearest_donors(request, donors, max_distance=50.0, max_results=10)
    
    print(f"Found {len(matches)} compatible donors:\n")
    
    for i, match in enumerate(matches, 1):
        print(f"{i}. {match.donor.full_name} ({match.donor.blood_group})")
        print(f"   Distance: {match.distance_km} km")
        print(f"   Overall Score: {match.overall_score}")
        print(f"   Compatibility Score: {match.compatibility_score}")
        print(f"   Available: {match.donor.available_date} ({match.donor.available_time})")
        print(f"   Contact: {match.donor.email}, {match.donor.phone}")
        print(f"   Address: {match.donor.formatted_address}")
        print()
    
    # Show statistics
    stats = matcher.get_matching_statistics(matches)
    print("=== Matching Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    example_usage()
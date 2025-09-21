"""
Geocoding Service - Convert text locations to coordinates
Uses OpenStreetMap's Nominatim API (free and open source)
Python implementation for blood donation website
"""

import requests
import time
import math
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class CoordinateResult:
    """Data class to represent geocoding results"""
    latitude: float
    longitude: float
    display_name: str
    formatted_address: str
    raw_response: Dict[str, Any]


class GeocodingService:
    """
    Geocoding service to convert text locations to coordinates
    """
    
    def __init__(self):
        """Initialize the geocoding service"""
        self.cache = {}  # Cache to avoid repeated API calls
        self.base_url = 'https://nominatim.openstreetmap.org/search'
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BloodDonationWebsite/1.0'
        })
    
    def get_coordinates(self, location_text: str, location_type: str = 'general') -> CoordinateResult:
        """
        Convert text location to coordinates
        
        Args:
            location_text (str): Text description of location
            location_type (str): Type of location ('general' or 'hospital')
            
        Returns:
            CoordinateResult: Object containing coordinates and address info
            
        Raises:
            ValueError: If location_text is invalid
            Exception: If geocoding fails
        """
        # Validate input
        if not location_text or not isinstance(location_text, str):
            raise ValueError('Location text is required and must be a string')
        
        # Check cache first
        cache_key = f"{location_text.lower().strip()}_{location_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Build search query based on type
            search_query = location_text.strip()
            if location_type == 'hospital':
                search_query = f"{location_text} hospital"
            
            # Prepare API request parameters
            params = {
                'q': search_query,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            
            # Make API request
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data and len(data) > 0:
                result = CoordinateResult(
                    latitude=float(data[0]['lat']),
                    longitude=float(data[0]['lon']),
                    display_name=data[0]['display_name'],
                    formatted_address=self._format_address(data[0]),
                    raw_response=data[0]
                )
                
                # Cache the result
                self.cache[cache_key] = result
                return result
            else:
                raise Exception(f"Location not found: {location_text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Geocoding API failed: {str(e)}")
        except Exception as e:
            print(f"Geocoding error for location: {location_text}, Error: {str(e)}")
            raise
    
    def get_donor_coordinates(self, donor_location: str) -> CoordinateResult:
        """
        Get coordinates for donor location
        
        Args:
            donor_location (str): Donor's location text
            
        Returns:
            CoordinateResult: Donor location coordinates
        """
        return self.get_coordinates(donor_location, 'general')
    
    def get_hospital_coordinates(self, hospital_name: str) -> CoordinateResult:
        """
        Get coordinates for hospital location
        
        Args:
            hospital_name (str): Hospital name
            
        Returns:
            CoordinateResult: Hospital location coordinates
        """
        return self.get_coordinates(hospital_name, 'hospital')
    
    def _format_address(self, nominatim_result: Dict[str, Any]) -> str:
        """
        Format address from Nominatim response for better readability
        
        Args:
            nominatim_result (dict): Raw response from Nominatim API
            
        Returns:
            str: Formatted address string
        """
        address = nominatim_result.get('address', {})
        parts = []
        
        # Build formatted address from address components
        if address.get('house_number'):
            parts.append(address['house_number'])
        if address.get('road'):
            parts.append(address['road'])
        if address.get('suburb') or address.get('neighbourhood'):
            parts.append(address.get('suburb') or address.get('neighbourhood'))
        if address.get('city') or address.get('town') or address.get('village'):
            parts.append(address.get('city') or address.get('town') or address.get('village'))
        if address.get('state'):
            parts.append(address['state'])
        if address.get('postcode'):
            parts.append(address['postcode'])
        if address.get('country'):
            parts.append(address['country'])
        
        return ', '.join(parts) if parts else nominatim_result['display_name']
    
    def get_multiple_coordinates(self, locations: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Batch geocoding for multiple locations
        
        Args:
            locations (list): List of location dictionaries with 'text' and 'type' keys
            
        Returns:
            list: List of geocoding results with success/failure status
        """
        results = []
        
        for location in locations:
            try:
                coords = self.get_coordinates(location['text'], location.get('type', 'general'))
                results.append({
                    'input': location,
                    'success': True,
                    'coordinates': coords
                })
            except Exception as error:
                results.append({
                    'input': location,
                    'success': False,
                    'error': str(error)
                })
            
            # Add small delay to respect API rate limits
            time.sleep(0.1)
        
        return results
    
    def is_valid_coordinates(self, coords: CoordinateResult) -> bool:
        """
        Validate if coordinates are valid
        
        Args:
            coords (CoordinateResult): Coordinates to validate
            
        Returns:
            bool: True if coordinates are valid, False otherwise
        """
        return (coords and
                isinstance(coords.latitude, (int, float)) and
                isinstance(coords.longitude, (int, float)) and
                -90 <= coords.latitude <= 90 and
                -180 <= coords.longitude <= 180)
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinate points using Haversine formula
        
        Args:
            lat1 (float): Latitude of first point
            lon1 (float): Longitude of first point
            lat2 (float): Latitude of second point
            lon2 (float): Longitude of second point
            
        Returns:
            float: Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers
        d_lat = self._to_radians(lat2 - lat1)
        d_lon = self._to_radians(lon2 - lon1)
        
        a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) +
             math.cos(self._to_radians(lat1)) * math.cos(self._to_radians(lat2)) *
             math.sin(d_lon / 2) * math.sin(d_lon / 2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c  # Distance in kilometers
    
    def _to_radians(self, degrees: float) -> float:
        """
        Convert degrees to radians
        
        Args:
            degrees (float): Angle in degrees
            
        Returns:
            float: Angle in radians
        """
        return degrees * (math.pi / 180)
    
    def clear_cache(self) -> None:
        """Clear geocoding cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            dict: Cache statistics including size and keys
        """
        return {
            'size': len(self.cache),
            'keys': list(self.cache.keys())
        }
    
    def set_base_url(self, url: str) -> None:
        """
        Set custom base URL for different geocoding services
        
        Args:
            url (str): New base URL for geocoding API
        """
        self.base_url = url
    
    def close(self) -> None:
        """Close the requests session"""
        if hasattr(self, 'session'):
            self.session.close()


class AsyncGeocodingService:
    """
    Asynchronous version of the geocoding service for better performance
    """
    
    def __init__(self):
        """Initialize the async geocoding service"""
        self.cache = {}
        self.base_url = 'https://nominatim.openstreetmap.org/search'
    
    async def get_coordinates_async(self, location_text: str, location_type: str = 'general') -> CoordinateResult:
        """
        Asynchronously convert text location to coordinates
        
        Args:
            location_text (str): Text description of location
            location_type (str): Type of location ('general' or 'hospital')
            
        Returns:
            CoordinateResult: Object containing coordinates and address info
        """
        # Validate input
        if not location_text or not isinstance(location_text, str):
            raise ValueError('Location text is required and must be a string')
        
        # Check cache first
        cache_key = f"{location_text.lower().strip()}_{location_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Build search query
            search_query = location_text.strip()
            if location_type == 'hospital':
                search_query = f"{location_text} hospital"
            
            params = {
                'q': search_query,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            
            # Make async API request
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {'User-Agent': 'BloodDonationWebsite/1.0'}
                async with session.get(self.base_url, params=params, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            if data and len(data) > 0:
                result = CoordinateResult(
                    latitude=float(data[0]['lat']),
                    longitude=float(data[0]['lon']),
                    display_name=data[0]['display_name'],
                    formatted_address=self._format_address(data[0]),
                    raw_response=data[0]
                )
                
                # Cache the result
                self.cache[cache_key] = result
                return result
            else:
                raise Exception(f"Location not found: {location_text}")
                
        except Exception as e:
            print(f"Async geocoding error for location: {location_text}, Error: {str(e)}")
            raise
    
    def _format_address(self, nominatim_result: Dict[str, Any]) -> str:
        """Format address from Nominatim response"""
        address = nominatim_result.get('address', {})
        parts = []
        
        if address.get('house_number'):
            parts.append(address['house_number'])
        if address.get('road'):
            parts.append(address['road'])
        if address.get('suburb') or address.get('neighbourhood'):
            parts.append(address.get('suburb') or address.get('neighbourhood'))
        if address.get('city') or address.get('town') or address.get('village'):
            parts.append(address.get('city') or address.get('town') or address.get('village'))
        if address.get('state'):
            parts.append(address['state'])
        if address.get('postcode'):
            parts.append(address['postcode'])
        if address.get('country'):
            parts.append(address['country'])
        
        return ', '.join(parts) if parts else nominatim_result['display_name']


def example_usage():
    """Example usage of the geocoding service"""
    geocoder = GeocodingService()
    
    print("=== Python Geocoding Service Examples ===\n")
    
    try:
        # Example 1: Get donor location coordinates
        print("1. Getting donor location coordinates:")
        donor_location = geocoder.get_donor_coordinates("Mumbai, Maharashtra, India")
        print(f"Donor location: {donor_location.latitude}, {donor_location.longitude}")
        print(f"Address: {donor_location.formatted_address}")
        print()

        # Example 2: Get hospital coordinates
        print("2. Getting hospital coordinates:")
        hospital_location = geocoder.get_hospital_coordinates("KEM Hospital Mumbai")
        print(f"Hospital location: {hospital_location.latitude}, {hospital_location.longitude}")
        print(f"Address: {hospital_location.formatted_address}")
        print()

        # Example 3: Calculate distance between locations
        print("3. Calculating distance:")
        distance = geocoder.calculate_distance(
            donor_location.latitude, donor_location.longitude,
            hospital_location.latitude, hospital_location.longitude
        )
        print(f"Distance: {distance:.2f} km")
        print()

        # Example 4: Batch geocoding
        print("4. Batch geocoding:")
        locations = [
            {'text': 'Pune, Maharashtra', 'type': 'general'},
            {'text': 'AIIMS Delhi', 'type': 'hospital'},
            {'text': 'Bangalore, Karnataka', 'type': 'general'}
        ]
        
        batch_results = geocoder.get_multiple_coordinates(locations)
        for i, result in enumerate(batch_results, 1):
            print(f"Location {i}: {'Success' if result['success'] else 'Failed'}")
            if result['success']:
                coords = result['coordinates']
                print(f"  Coordinates: {coords.latitude}, {coords.longitude}")
            else:
                print(f"  Error: {result['error']}")
        print()

        # Example 5: Cache statistics
        print("5. Cache statistics:")
        print(geocoder.get_cache_stats())

    except Exception as error:
        print(f"Example error: {str(error)}")
    
    finally:
        # Clean up
        geocoder.close()


if __name__ == "__main__":
    example_usage()
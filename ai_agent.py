"""
AI Agent for Blood Donation Automation
Uses LangChain agents and LangGraph pipeline with OpenAI-compatible APIs
Automates geocoding, donor matching, and email generation
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from concurrent.futures import ThreadPoolExecutor

# LangChain imports (updated for compatibility)
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI

# LangGraph imports (updated for compatibility)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict

# Import our custom modules (handle import errors gracefully)
try:
    from geocoding_service import GeocodingService, CoordinateResult
    from donor_matching import DonorMatchingEngine, Donor, BloodRequest, MatchResult
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    raise ImportError("Please ensure geocoding_service.py and donor_matching.py are in the same directory")


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BloodDonationState(TypedDict):
    """State object for the blood donation workflow"""
    messages: List[Dict[str, Any]]
    donor_data: Optional[Dict[str, Any]]
    request_data: Optional[Dict[str, Any]]
    donor_coordinates: Optional[Dict[str, float]]
    hospital_coordinates: Optional[Dict[str, float]]
    matching_results: Optional[List[Dict[str, Any]]]
    email_notifications: Optional[List[Dict[str, Any]]]
    current_step: str
    error_message: Optional[str]
    success: bool


@dataclass
class EmailConfig:
    """Email configuration for notifications"""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_email: str = ""


class BloodDonationTools:
    """Tools for the blood donation AI agent"""
    
    def __init__(self):
        self.geocoder = GeocodingService()
        self.matcher = DonorMatchingEngine()
        self.donors_database = []  # In-memory donor storage
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _convert_coord_result_to_dict(self, coord_result: CoordinateResult) -> Dict[str, Any]:
        """Convert CoordinateResult to dictionary"""
        return {
            "latitude": coord_result.latitude,
            "longitude": coord_result.longitude,
            "formatted_address": coord_result.formatted_address,
            "display_name": coord_result.display_name
        }
    
    def _validate_donor_data(self, donor_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate donor data"""
        required_fields = ['full_name', 'email', 'phone', 'blood_group', 'location', 'available_date']
        
        for field in required_fields:
            if field not in donor_data or not donor_data[field]:
                return False, f"Missing required field: {field}"
        
        # Validate blood group
        valid_blood_groups = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
        if donor_data['blood_group'] not in valid_blood_groups:
            return False, f"Invalid blood group: {donor_data['blood_group']}"
        
        # Validate date format
        try:
            datetime.strptime(donor_data['available_date'], "%Y-%m-%d")
        except ValueError:
            return False, "Invalid date format. Use YYYY-MM-DD"
        
        return True, ""
    
    def _validate_request_data(self, request_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate blood request data"""
        required_fields = ['patient_name', 'contact_email', 'contact_phone', 'blood_group', 
                          'hospital_name', 'urgency', 'units_needed', 'required_date']
        
        for field in required_fields:
            if field not in request_data or not request_data[field]:
                return False, f"Missing required field: {field}"
        
        # Validate blood group
        valid_blood_groups = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
        if request_data['blood_group'] not in valid_blood_groups:
            return False, f"Invalid blood group: {request_data['blood_group']}"
        
        # Validate urgency
        valid_urgency = ["low", "normal", "high", "critical"]
        if request_data['urgency'].lower() not in valid_urgency:
            return False, f"Invalid urgency level: {request_data['urgency']}"
        
        # Validate date format
        try:
            datetime.strptime(request_data['required_date'], "%Y-%m-%d")
        except ValueError:
            return False, "Invalid date format. Use YYYY-MM-DD"
        
        return True, ""
    
    def geocode_donor_location(self, location_text: str) -> str:
        """
        Tool to geocode donor location
        
        Args:
            location_text: Text description of donor location
            
        Returns:
            str: JSON string with coordinates or error message
        """
        try:
            if not location_text or not isinstance(location_text, str):
                raise ValueError("Location text is required and must be a string")
            
            result = self.geocoder.get_donor_coordinates(location_text)
            return json.dumps({
                "success": True,
                "latitude": result.latitude,
                "longitude": result.longitude,
                "formatted_address": result.formatted_address,
                "display_name": result.display_name
            })
        except Exception as e:
            logger.error(f"Geocoding error for donor location: {location_text}, Error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def _geocode_donor_location_internal(self, location_text: str) -> CoordinateResult:
        """Internal method for geocoding donor location (returns object)"""
        return self.geocoder.get_donor_coordinates(location_text)
    
    def geocode_hospital_location(self, hospital_name: str) -> str:
        """
        Tool to geocode hospital location
        
        Args:
            hospital_name: Name of the hospital
            
        Returns:
            str: JSON string with coordinates or error message
        """
        try:
            if not hospital_name or not isinstance(hospital_name, str):
                raise ValueError("Hospital name is required and must be a string")
            
            result = self.geocoder.get_hospital_coordinates(hospital_name)
            return json.dumps({
                "success": True,
                "latitude": result.latitude,
                "longitude": result.longitude,
                "formatted_address": result.formatted_address,
                "display_name": result.display_name
            })
        except Exception as e:
            logger.error(f"Geocoding error for hospital: {hospital_name}, Error: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def _geocode_hospital_location_internal(self, hospital_name: str) -> CoordinateResult:
        """Internal method for geocoding hospital location (returns object)"""
        return self.geocoder.get_hospital_coordinates(hospital_name)
    
    def add_donor_to_database(self, donor_data: str) -> str:
        """
        Tool to add donor to database
        
        Args:
            donor_data: JSON string with donor information
            
        Returns:
            str: Success/failure message
        """
        try:
            data = json.loads(donor_data)
            
            # Validate donor data
            is_valid, error_msg = self._validate_donor_data(data)
            if not is_valid:
                raise ValueError(error_msg)
            
            donor = Donor(
                id=f"D{len(self.donors_database) + 1:03d}",
                full_name=data['full_name'],
                email=data['email'],
                phone=data['phone'],
                blood_group=data['blood_group'],
                latitude=data['latitude'],
                longitude=data['longitude'],
                location_text=data['location_text'],
                formatted_address=data['formatted_address'],
                available_date=data['available_date'],
                available_time=data['available_time'],
                is_active=True,
                created_at=datetime.now()
            )
            self.donors_database.append(donor)
            
            logger.info(f"Donor {donor.full_name} added successfully with ID {donor.id}")
            
            return json.dumps({
                "success": True,
                "message": f"Donor {donor.full_name} added successfully with ID {donor.id}",
                "donor_id": donor.id
            })
        except Exception as e:
            logger.error(f"Error adding donor to database: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def find_matching_donors(self, request_data: str, max_distance: int = 50, max_results: int = 10) -> str:
        """
        Tool to find matching donors for a blood request
        
        Args:
            request_data: JSON string with blood request information
            max_distance: Maximum search distance in km
            max_results: Maximum number of results
            
        Returns:
            str: JSON string with matching results
        """
        try:
            data = json.loads(request_data)
            
            # Validate request data
            is_valid, error_msg = self._validate_request_data(data)
            if not is_valid:
                raise ValueError(error_msg)
            
            # Create BloodRequest object
            request = BloodRequest(
                id=f"R{datetime.now().strftime('%Y%m%d%H%M%S')}",
                patient_name=data['patient_name'],
                contact_email=data['contact_email'],
                contact_phone=data['contact_phone'],
                blood_group=data['blood_group'],
                hospital_name=data['hospital_name'],
                latitude=data['latitude'],
                longitude=data['longitude'],
                hospital_address=data['hospital_address'],
                urgency=data.get('urgency', 'normal'),
                units_needed=int(data.get('units_needed', 1)),
                required_date=data['required_date'],
                created_at=datetime.now()
            )
            
            # Find matching donors
            matches = self.matcher.find_nearest_donors(
                request=request,
                donors=self.donors_database,
                max_distance=max_distance,
                max_results=max_results
            )
            
            # Convert to serializable format
            results = []
            for match in matches:
                results.append({
                    "donor_id": match.donor.id,
                    "donor_name": match.donor.full_name,
                    "donor_email": match.donor.email,
                    "donor_phone": match.donor.phone,
                    "donor_blood_group": match.donor.blood_group,
                    "donor_address": match.donor.formatted_address,
                    "distance_km": match.distance_km,
                    "overall_score": match.overall_score,
                    "compatibility_score": match.compatibility_score,
                    "available_date": match.donor.available_date,
                    "available_time": match.donor.available_time
                })
            
            stats = self.matcher.get_matching_statistics(matches)
            
            logger.info(f"Found {len(results)} matching donors for request {request.id}")
            
            return json.dumps({
                "success": True,
                "total_matches": len(results),
                "matches": results,
                "statistics": stats,
                "request_id": request.id
            })
        except Exception as e:
            logger.error(f"Error finding matching donors: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def generate_notification_emails(self, matches_data: str, request_data: str) -> str:
        """
        Tool to generate email notifications for matched donors
        
        Args:
            matches_data: JSON string with matching results
            request_data: JSON string with request information
            
        Returns:
            str: JSON string with email content
        """
        try:
            matches = json.loads(matches_data)
            request = json.loads(request_data)
            
            if not matches.get('success', False):
                raise ValueError("Invalid matches data provided")
            
            emails = []
            for match in matches.get('matches', []):
                email_content = self._create_email_content(match, request)
                emails.append(email_content)
            
            logger.info(f"Generated {len(emails)} email notifications")
            
            return json.dumps({
                "success": True,
                "emails_generated": len(emails),
                "emails": emails
            })
        except Exception as e:
            logger.error(f"Error generating email notifications: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def _create_email_content(self, donor_match: Dict, request_data: Dict) -> Dict[str, str]:
        """Create email content for donor notification"""
        subject = f"Urgent Blood Donation Request - {request_data['blood_group']} Blood Needed"
        
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #d73027;">Blood Donation Request</h2>
            
            <p>Dear <strong>{donor_match['donor_name']}</strong>,</p>
            
            <p>We have received an urgent blood donation request that matches your profile. Your help could save a life!</p>
            
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <h3 style="color: #333; margin-top: 0;">Request Details:</h3>
                <p><strong>Patient:</strong> {request_data['patient_name']}</p>
                <p><strong>Blood Group Required:</strong> {request_data['blood_group']}</p>
                <p><strong>Hospital:</strong> {request_data['hospital_name']}</p>
                <p><strong>Required Date:</strong> {request_data['required_date']}</p>
                <p><strong>Units Needed:</strong> {request_data.get('units_needed', 1)}</p>
                <p><strong>Urgency:</strong> {request_data.get('urgency', 'Normal')}</p>
            </div>
            
            <div style="background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <h3 style="color: #333; margin-top: 0;">Contact Information:</h3>
                <p><strong>Contact Person:</strong> {request_data['contact_email']}</p>
                <p><strong>Phone:</strong> {request_data['contact_phone']}</p>
            </div>
            
            <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <h3 style="color: #333; margin-top: 0;">Location Details:</h3>
                <p><strong>Distance from you:</strong> {donor_match['distance_km']} km</p>
                <p><strong>Your availability:</strong> {donor_match['available_date']} ({donor_match['available_time']})</p>
                <p><strong>Match Score:</strong> {donor_match['overall_score']}/100</p>
            </div>
            
            <p><strong>If you are available and willing to donate, please contact the requester directly using the contact information provided above.</strong></p>
            
            <p style="color: #666; font-size: 14px;">
                Thank you for being a registered blood donor. Your generosity can make a life-changing difference.
            </p>
            
            <hr style="margin: 30px 0;">
            <p style="color: #999; font-size: 12px;">
                This is an automated message from the Blood Donation AI System.
            </p>
        </div>
        """
        
        return {
            "to_email": donor_match['donor_email'],
            "to_name": donor_match['donor_name'],
            "subject": subject,
            "html_content": html_content
        }
    
    def get_donor_count(self) -> int:
        """Get total number of registered donors"""
        return len(self.donors_database)
    
    def get_active_donor_count(self) -> int:
        """Get number of active donors"""
        return sum(1 for donor in self.donors_database if donor.is_active)
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'geocoder'):
            self.geocoder.close()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class BloodDonationAgent:
    """AI Agent for blood donation automation"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.tools_instance = BloodDonationTools()
        
        # Initialize LLM with flexible configuration
        llm_config = {
            "api_key": api_key,
            "model": model,
            "temperature": 0.1,
            "max_tokens": 2000,
            "timeout": 30
        }
        
        if base_url:
            llm_config["base_url"] = base_url
        
        self.llm = ChatOpenAI(**llm_config)
        
        # Create tools for the agent
        self.tools = self._create_tools()
        
        # Create the agent
        self.agent_executor = self._create_agent()
        
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
    
    def _create_tools(self):
        """Create tools for the agent"""

        @tool
        def geocode_donor_location(location_text: str) -> str:
            """Geocode donor location text to coordinates. Input should be location text string."""
            return self.tools_instance.geocode_donor_location(location_text)

        @tool
        def geocode_hospital_location(hospital_name: str) -> str:
            """Geocode hospital name to coordinates. Input should be hospital name string."""
            return self.tools_instance.geocode_hospital_location(hospital_name)

        @tool
        def add_donor_to_database(donor_data: str) -> str:
            """Add donor to database. Input should be JSON string with donor data including coordinates."""
            return self.tools_instance.add_donor_to_database(donor_data)

        @tool
        def find_matching_donors(request_data: str, max_distance: int = 50, max_results: int = 10) -> str:
            """Find matching donors for blood request. Input should be JSON string with request data including coordinates."""
            return self.tools_instance.find_matching_donors(request_data, max_distance, max_results)

        @tool
        def generate_notification_emails(matches_data: str, request_data: str) -> str:
            """Generate email notifications for matched donors. Takes matches data and request data as JSON strings."""
            return self.tools_instance.generate_notification_emails(matches_data, request_data)

        return [geocode_donor_location, geocode_hospital_location, add_donor_to_database,
                find_matching_donors, generate_notification_emails]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the OpenAI Functions agent"""

        # Bind tools to the prompt
        if self.tools:
            tools_description = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        else:
            tools_description = "No tools available"

        system_prompt = f"""You are an AI assistant specialized in blood donation management. You help automate the process of:
        1. Registering donors with location geocoding
        2. Processing blood requests with hospital location geocoding
        3. Finding matching donors based on proximity and compatibility
        4. Generating email notifications for matched donors

        You have access to the following tools: {tools_description}

        Use the tools available to help users with blood donation tasks."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        prompt_with_tools = prompt.partial(tools=tools_description or "No tools available")

        agent = create_openai_functions_agent(self.llm, self.tools, prompt_with_tools)
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
    
    def _create_workflow(self):
        """Create LangGraph workflow for the blood donation process"""
        
        workflow = StateGraph(BloodDonationState)
        
        # Add nodes
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("geocode_locations", self._geocode_locations)
        workflow.add_node("find_matches", self._find_matches)
        workflow.add_node("generate_emails", self._generate_emails)
        workflow.add_node("finalize", self._finalize)
        
        # Add edges
        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "geocode_locations")
        workflow.add_edge("geocode_locations", "find_matches")
        workflow.add_edge("find_matches", "generate_emails")
        workflow.add_edge("generate_emails", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def _process_input(self, state: BloodDonationState) -> BloodDonationState:
        """Process initial input and determine workflow type"""
        try:
            messages = state.get("messages", [])
            if not messages:
                raise ValueError("No messages found in state")

            last_message = messages[-1]
            content = last_message.get("content", {})

            if content and "donor" in content:
                state["donor_data"] = content["donor"]
                state["current_step"] = "donor_registration"
            elif content and "request" in content:
                state["request_data"] = content["request"]
                state["current_step"] = "blood_request"
            else:
                raise ValueError("Invalid input: must contain either 'donor' or 'request' data")

            state["error_message"] = None
            return state
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            state["error_message"] = str(e)
            state["success"] = False
            return state
    
    async def _geocode_locations(self, state: BloodDonationState) -> BloodDonationState:
        """Geocode locations using internal methods"""
        try:
            if state.get("donor_data") and state["donor_data"]:
                # Geocode donor location
                location_text = state["donor_data"]["location"]
                coords = await asyncio.get_event_loop().run_in_executor(
                    self.tools_instance.executor,
                    self.tools_instance._geocode_donor_location_internal,
                    location_text
                )
                state["donor_coordinates"] = self.tools_instance._convert_coord_result_to_dict(coords)
                logger.info(f"Geocoded donor location: {location_text}")
            
            if state.get("request_data") and state["request_data"]:
                # Geocode hospital location
                hospital_name = state["request_data"]["hospital_name"]
                coords = await asyncio.get_event_loop().run_in_executor(
                    self.tools_instance.executor,
                    self.tools_instance._geocode_hospital_location_internal,
                    hospital_name
                )
                state["hospital_coordinates"] = self.tools_instance._convert_coord_result_to_dict(coords)
                logger.info(f"Geocoded hospital location: {hospital_name}")
            
            return state
        except Exception as e:
            logger.error(f"Error geocoding locations: {e}")
            state["error_message"] = str(e)
            state["success"] = False
            return state
    
    async def _find_matches(self, state: BloodDonationState) -> BloodDonationState:
        """Find matching donors"""
        try:
            request_data = state.get("request_data")
            hospital_coordinates = state.get("hospital_coordinates")

            if request_data and hospital_coordinates:
                # Prepare request data with coordinates
                request_data = {
                    **request_data,
                    **hospital_coordinates,
                    "hospital_address": hospital_coordinates["formatted_address"]
                }

                # Use internal method to find matches
                matches_json = await asyncio.get_event_loop().run_in_executor(
                    self.tools_instance.executor,
                    self.tools_instance.find_matching_donors,
                    json.dumps(request_data)
                )

                matches_data = json.loads(matches_json)

                if matches_data["success"]:
                    state["matching_results"] = matches_data
                    logger.info(f"Found {matches_data['total_matches']} matching donors")
                else:
                    raise Exception(matches_data["error"])

            return state
        except Exception as e:
            logger.error(f"Error finding matches: {e}")
            state["error_message"] = str(e)
            state["success"] = False
            return state
    
    async def _generate_emails(self, state: BloodDonationState) -> BloodDonationState:
        """Generate email notifications"""
        try:
            if state.get("matching_results") and state.get("request_data"):
                # Use internal method to generate emails
                email_json = await asyncio.get_event_loop().run_in_executor(
                    self.tools_instance.executor,
                    self.tools_instance.generate_notification_emails,
                    json.dumps(state["matching_results"]),
                    json.dumps(state["request_data"])
                )
                
                email_data = json.loads(email_json)
                
                if email_data["success"]:
                    state["email_notifications"] = email_data
                    logger.info(f"Generated {email_data['emails_generated']} email notifications")
                else:
                    raise Exception(email_data["error"])
            
            return state
        except Exception as e:
            logger.error(f"Error generating emails: {e}")
            state["error_message"] = str(e)
            state["success"] = False
            return state
    
    async def _finalize(self, state: BloodDonationState) -> BloodDonationState:
        """Finalize the workflow"""
        if not state.get("error_message"):
            state["success"] = True
            state["current_step"] = "completed"
            logger.info("Workflow completed successfully")
        else:
            logger.error(f"Workflow failed: {state['error_message']}")
        
        return state
    
    async def register_donor(self, donor_data: Dict[str, str]) -> Dict[str, Any]:
        """Register a new donor with automated geocoding"""
        try:
            logger.info(f"Registering donor: {donor_data.get('full_name', 'Unknown')}")
            
            # Validate input data
            is_valid, error_msg = self.tools_instance._validate_donor_data(donor_data)
            if not is_valid:
                return {"success": False, "error": error_msg}
            
            # Geocode location
            coords = await asyncio.get_event_loop().run_in_executor(
                self.tools_instance.executor,
                self.tools_instance._geocode_donor_location_internal,
                donor_data["location"]
            )
            
            # Add coordinates to donor data
            complete_donor_data = {
                **donor_data,
                "latitude": coords.latitude,
                "longitude": coords.longitude,
                "formatted_address": coords.formatted_address,
                "location_text": donor_data["location"]
            }
            
            # Add to database
            add_result = json.loads(
                self.tools_instance.add_donor_to_database(json.dumps(complete_donor_data))
            )
            
            return add_result
        except Exception as e:
            logger.error(f"Error registering donor: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_blood_request(self, request_data: Dict[str, str]) -> Dict[str, Any]:
        """Process blood request and find matches"""
        try:
            logger.info(f"Processing blood request for patient: {request_data.get('patient_name', 'Unknown')}")
            
            # Validate input data
            is_valid, error_msg = self.tools_instance._validate_request_data(request_data)
            if not is_valid:
                return {"success": False, "error": error_msg}
            
            initial_state = BloodDonationState(
                messages=[{"content": {"request": request_data}}],
                donor_data=None,
                request_data=None,
                donor_coordinates=None,
                hospital_coordinates=None,
                matching_results=None,
                email_notifications=None,
                current_step="start",
                error_message=None,
                success=False
            )
            
            # Run workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            return {
                "success": final_state.get("success", False),
                "error": final_state.get("error_message"),
                "matches": final_state.get("matching_results"),
                "emails": final_state.get("email_notifications"),
                "current_step": final_state.get("current_step", "unknown")
            }
        except Exception as e:
            logger.error(f"Error processing blood request: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            return {
                "total_donors": self.tools_instance.get_donor_count(),
                "active_donors": self.tools_instance.get_active_donor_count(),
                "geocoding_cache_size": len(self.tools_instance.geocoder.cache),
                "system_status": "operational"
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'tools_instance'):
            self.tools_instance.close()


# Example usage and testing
async def main():
    """Example usage of the Blood Donation AI Agent"""
    
    # Initialize agent with API key
    API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "your-api-key-here"
    
    # For DeepSeek API, uncomment the following lines:
    # BASE_URL = "https://api.deepseek.com/v1"
    # MODEL = "deepseek-chat"
    # agent = BloodDonationAgent(API_KEY, base_url=BASE_URL, model=MODEL)
    
    # For OpenAI API (default):
    agent = BloodDonationAgent(API_KEY)
    
    print("=== Blood Donation AI Agent Demo ===\n")
    
    try:
        # Example 1: Register donors
        print("1. Registering donors...")
        
        donor1 = {
            "full_name": "John Doe",
            "email": "john.doe@email.com", 
            "phone": "+1234567890",
            "blood_group": "O+",
            "location": "Mumbai, Maharashtra, India",
            "available_date": "2024-03-15",
            "available_time": "09:00-17:00"
        }
        
        result1 = await agent.register_donor(donor1)
        print(f"Donor 1 registration: {result1}")
        
        donor2 = {
            "full_name": "Jane Smith",
            "email": "jane.smith@email.com",
            "phone": "+1234567891", 
            "blood_group": "A+",
            "location": "Andheri, Mumbai, Maharashtra",
            "available_date": "2024-03-16",
            "available_time": "10:00-16:00"
        }
        
        result2 = await agent.register_donor(donor2)
        print(f"Donor 2 registration: {result2}")
        
        donor3 = {
            "full_name": "Mike Johnson",
            "email": "mike.johnson@email.com",
            "phone": "+1234567892",
            "blood_group": "A+",
            "location": "Powai, Mumbai, Maharashtra", 
            "available_date": "2024-03-16",
            "available_time": "08:00-18:00"
        }
        
        result3 = await agent.register_donor(donor3)
        print(f"Donor 3 registration: {result3}")
        print()
        
        # Example 2: Process blood request
        print("2. Processing blood request...")
        
        request = {
            "patient_name": "Emergency Patient",
            "contact_email": "hospital@kem.com",
            "contact_phone": "+1234567892",
            "blood_group": "A+", 
            "hospital_name": "KEM Hospital Mumbai",
            "urgency": "high",
            "units_needed": "2",
            "required_date": "2024-03-16"
        }
        
        request_result = await agent.process_blood_request(request)
        print(f"Request processing result: {request_result['success']}")
        
        if request_result["success"] and request_result["matches"]:
            print(f"Found {request_result['matches']['total_matches']} matching donors")
            print(f"Email notifications generated: {request_result['emails']['emails_generated']}")
            
            # Display first few matches
            matches = request_result['matches']['matches'][:3]
            print("\nTop matching donors:")
            for i, match in enumerate(matches, 1):
                print(f"{i}. {match['donor_name']} ({match['donor_blood_group']})")
                print(f"   Distance: {match['distance_km']} km")
                print(f"   Score: {match['overall_score']}")
                print(f"   Contact: {match['donor_email']}")
        
        # Example 3: System stats
        print("\n3. System Statistics:")
        stats = await agent.get_system_stats()
        print(f"Total donors: {stats.get('total_donors', 0)}")
        print(f"Active donors: {stats.get('active_donors', 0)}")
        print(f"System status: {stats.get('system_status', 'unknown')}")
        
    except Exception as e:
        print(f"Demo error: {e}")
    
    finally:
        # Clean up
        agent.close()


if __name__ == "__main__":
    asyncio.run(main())
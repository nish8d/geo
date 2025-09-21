# AI Integration Guide for Blood Donation Website

## Overview

This document provides comprehensive instructions for integrating the AI-powered blood donation system into your main website project. The AI system consists of three main components that work together to automate donor matching, location geocoding, and workflow orchestration.

## 📁 AI System Components

### 1. **ai_agent.py** - Main AI Orchestrator
- **Purpose**: Central AI agent using LangChain and LangGraph
- **Features**:
  - Automated donor registration with geocoding
  - Blood request processing with hospital location detection
  - Intelligent donor matching based on proximity and compatibility
  - Automated email notification generation
  - Workflow state management

### 2. **geocoding_service.py** - Location Intelligence
- **Purpose**: Convert text locations to coordinates
- **Features**:
  - OpenStreetMap Nominatim API integration
  - Donor and hospital location geocoding
  - Distance calculation between locations
  - Caching for improved performance
  - Batch processing capabilities

### 3. **donor_matching.py** - Smart Matching Algorithm
- **Purpose**: Find compatible donors for blood requests
- **Features**:
  - Blood type compatibility matrix
  - Distance-based proximity scoring
  - Urgency level prioritization
  - Availability date matching
  - Comprehensive scoring system (0-100)

## 🚀 Integration Instructions

### Step 1: Project Structure Setup

Create the following directory structure in your main project:

```
your-blood-donation-website/
├── ai/
│   ├── __init__.py
│   ├── ai_agent.py
│   ├── geocoding_service.py
│   └── donor_matching.py
├── requirements.txt
├── app.py (or your main application file)
└── ...
```

### Step 2: Install Dependencies

Update your `requirements.txt` file with the following dependencies:

```txt
# Core AI and ML dependencies
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.13
langgraph>=0.0.20

# HTTP and API dependencies
requests>=2.31.0
aiohttp>=3.9.1

# Data processing
python-dateutil>=2.8.2
dataclasses>=0.6

# Optional: For enhanced functionality
python-dotenv>=1.0.0
pydantic>=2.5.0
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Environment Configuration

Create a `.env` file in your project root:

```env
# OpenAI API Configuration (choose one)
OPENAI_API_KEY=your_openai_api_key_here
# OR for DeepSeek
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Email Configuration (for notifications)
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_FROM_EMAIL=your_email@gmail.com

# Application Settings
FLASK_ENV=development
DEBUG=True
```

### Step 4: Integration with Your Main Application

#### Option A: Flask Integration

If you're using Flask, add this to your main `app.py`:

```python
# Import AI components
from ai.ai_agent import BloodDonationAgent
from ai.geocoding_service import GeocodingService
from ai.donor_matching import DonorMatchingEngine

# Initialize AI components
ai_agent = None
geocoder = GeocodingService()
matcher = DonorMatchingEngine()

def get_ai_agent():
    """Get or create AI agent instance"""
    global ai_agent
    if ai_agent is None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            ai_agent = BloodDonationAgent(api_key)
    return ai_agent

# Add to your Flask app initialization
app = Flask(__name__)

# Make AI components available to routes
app.geocoder = geocoder
app.matcher = matcher
app.get_ai_agent = get_ai_agent
```

#### Option B: Django Integration

For Django, add to your `settings.py`:

```python
# Add to INSTALLED_APPS
INSTALLED_APPS = [
    # ... your existing apps
    'ai_integration',
]

# Add to your main urls.py
from django.urls import path, include

urlpatterns = [
    # ... your existing URLs
    path('ai/', include('ai_integration.urls')),
]
```

### Step 5: API Endpoints Integration

Create new API endpoints in your application:

#### Donor Registration Endpoint

```python
@app.route('/api/ai/register-donor', methods=['POST'])
async def register_donor():
    """Register a new donor with AI-powered geocoding"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['full_name', 'email', 'phone', 'blood_group', 'location']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Get AI agent
        agent = get_ai_agent()
        if not agent:
            return jsonify({'error': 'AI service not available'}), 503

        # Register donor
        result = await agent.register_donor(data)

        if result.get('success'):
            return jsonify({
                'message': 'Donor registered successfully',
                'donor_id': result.get('donor_id'),
                'coordinates': result.get('coordinates', {})
            }), 201
        else:
            return jsonify({'error': result.get('error')}), 400

    except Exception as e:
        logger.error(f"Donor registration error: {e}")
        return jsonify({'error': 'Internal server error'}), 500
```

#### Blood Request Processing Endpoint

```python
@app.route('/api/ai/process-request', methods=['POST'])
async def process_blood_request():
    """Process blood request with AI matching"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['patient_name', 'contact_email', 'blood_group',
                          'hospital_name', 'urgency', 'units_needed', 'required_date']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Get AI agent
        agent = get_ai_agent()
        if not agent:
            return jsonify({'error': 'AI service not available'}), 503

        # Process request
        result = await agent.process_blood_request(data)

        if result.get('success'):
            return jsonify({
                'message': 'Blood request processed successfully',
                'request_id': result.get('request_id'),
                'matches': result.get('matches', {}),
                'emails': result.get('emails', {})
            }), 200
        else:
            return jsonify({'error': result.get('error')}), 400

    except Exception as e:
        logger.error(f"Blood request processing error: {e}")
        return jsonify({'error': 'Internal server error'}), 500
```

#### Location Geocoding Endpoint

```python
@app.route('/api/ai/geocode', methods=['POST'])
def geocode_location():
    """Geocode location text to coordinates"""
    try:
        data = request.get_json()

        if 'location' not in data:
            return jsonify({'error': 'Location text is required'}), 400

        location_type = data.get('type', 'general')  # 'general' or 'hospital'

        # Use geocoding service
        if location_type == 'hospital':
            result = app.geocoder.get_hospital_coordinates(data['location'])
        else:
            result = app.geocoder.get_donor_coordinates(data['location'])

        return jsonify({
            'latitude': result.latitude,
            'longitude': result.longitude,
            'formatted_address': result.formatted_address,
            'display_name': result.display_name
        }), 200

    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return jsonify({'error': str(e)}), 400
```

### Step 6: Frontend Integration

#### JavaScript Integration

Create a new JavaScript file for AI functionality:

```javascript
// ai-service.js
class AIService {
    constructor(baseUrl = '/api/ai') {
        this.baseUrl = baseUrl;
    }

    async registerDonor(donorData) {
        try {
            const response = await fetch(`${this.baseUrl}/register-donor`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(donorData)
            });

            return await response.json();
        } catch (error) {
            console.error('Donor registration error:', error);
            throw error;
        }
    }

    async processBloodRequest(requestData) {
        try {
            const response = await fetch(`${this.baseUrl}/process-request`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            return await response.json();
        } catch (error) {
            console.error('Blood request processing error:', error);
            throw error;
        }
    }

    async geocodeLocation(locationText, type = 'general') {
        try {
            const response = await fetch(`${this.baseUrl}/geocode`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    location: locationText,
                    type: type
                })
            });

            return await response.json();
        } catch (error) {
            console.error('Geocoding error:', error);
            throw error;
        }
    }
}

// Export for use in your application
window.AIService = AIService;
```

#### HTML Integration Example

```html
<!-- Add to your donor registration form -->
<form id="donorRegistrationForm">
    <div class="form-group">
        <label for="fullName">Full Name</label>
        <input type="text" id="fullName" name="full_name" required>
    </div>

    <div class="form-group">
        <label for="email">Email</label>
        <input type="email" id="email" name="email" required>
    </div>

    <div class="form-group">
        <label for="phone">Phone</label>
        <input type="tel" id="phone" name="phone" required>
    </div>

    <div class="form-group">
        <label for="bloodGroup">Blood Group</label>
        <select id="bloodGroup" name="blood_group" required>
            <option value="">Select Blood Group</option>
            <option value="O+">O+</option>
            <option value="O-">O-</option>
            <option value="A+">A+</option>
            <option value="A-">A-</option>
            <option value="B+">B+</option>
            <option value="B-">B-</option>
            <option value="AB+">AB+</option>
            <option value="AB-">AB-</option>
        </select>
    </div>

    <div class="form-group">
        <label for="location">Location</label>
        <input type="text" id="location" name="location" placeholder="Enter your city, state" required>
        <button type="button" onclick="geocodeLocation()">Find Coordinates</button>
    </div>

    <div class="form-group">
        <label for="availableDate">Available Date</label>
        <input type="date" id="availableDate" name="available_date" required>
    </div>

    <div class="form-group">
        <label for="availableTime">Available Time</label>
        <input type="text" id="availableTime" name="available_time" placeholder="e.g., 09:00-17:00" required>
    </div>

    <button type="submit">Register as Donor</button>
</form>

<script>
const aiService = new AIService();

async function geocodeLocation() {
    const locationInput = document.getElementById('location');
    const location = locationInput.value;

    if (!location) {
        alert('Please enter a location first');
        return;
    }

    try {
        const result = await aiService.geocodeLocation(location, 'general');
        alert(`Coordinates found:\nLatitude: ${result.latitude}\nLongitude: ${result.longitude}\nAddress: ${result.formatted_address}`);
    } catch (error) {
        alert('Error geocoding location: ' + error.message);
    }
}

document.getElementById('donorRegistrationForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const donorData = Object.fromEntries(formData.entries());

    try {
        const result = await aiService.registerDonor(donorData);
        if (result.message) {
            alert('Success: ' + result.message);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
});
</script>
```

### Step 7: Database Integration

#### SQLAlchemy Models (if using SQLAlchemy)

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Donor(Base):
    __tablename__ = 'ai_donors'

    id = Column(String, primary_key=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False, unique=True)
    phone = Column(String(20), nullable=False)
    blood_group = Column(String(5), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    location_text = Column(Text)
    formatted_address = Column(Text)
    available_date = Column(String(20))
    available_time = Column(String(50))
    last_donation_date = Column(String(20))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class BloodRequest(Base):
    __tablename__ = 'ai_blood_requests'

    id = Column(String, primary_key=True)
    patient_name = Column(String(100), nullable=False)
    contact_email = Column(String(100), nullable=False)
    contact_phone = Column(String(20), nullable=False)
    blood_group = Column(String(5), nullable=False)
    hospital_name = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    hospital_address = Column(Text)
    urgency = Column(String(20), default='normal')
    units_needed = Column(Integer, default=1)
    required_date = Column(String(20), nullable=False)
    required_time = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
```

### Step 8: Testing the Integration

#### Create a test script:

```python
# test_ai_integration.py
import asyncio
import os
from ai.ai_agent import BloodDonationAgent

async def test_integration():
    """Test the AI integration"""

    # Initialize AI agent
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ API key not found. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY")
        return

    agent = BloodDonationAgent(api_key)

    try:
        # Test 1: Register donors
        print("🧪 Testing donor registration...")

        donor_data = {
            "full_name": "Test Donor",
            "email": "test@example.com",
            "phone": "+1234567890",
            "blood_group": "O+",
            "location": "Mumbai, Maharashtra, India",
            "available_date": "2024-03-20",
            "available_time": "09:00-17:00"
        }

        result = await agent.register_donor(donor_data)
        print(f"✅ Donor registration: {'Success' if result['success'] else 'Failed'}")

        # Test 2: Process blood request
        print("\n🧪 Testing blood request processing...")

        request_data = {
            "patient_name": "Test Patient",
            "contact_email": "hospital@test.com",
            "contact_phone": "+1234567891",
            "blood_group": "O+",
            "hospital_name": "Test Hospital Mumbai",
            "urgency": "high",
            "units_needed": 2,
            "required_date": "2024-03-20"
        }

        result = await agent.process_blood_request(request_data)
        print(f"✅ Request processing: {'Success' if result['success'] else 'Failed'}")

        if result['success']:
            matches = result.get('matches', {})
            print(f"📊 Found {matches.get('total_matches', 0)} matching donors")

        # Test 3: System statistics
        print("\n🧪 Testing system statistics...")
        stats = await agent.get_system_stats()
        print(f"✅ System stats: {stats}")

        print("\n🎉 All tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        agent.close()

if __name__ == "__main__":
    asyncio.run(test_integration())
```

Run the test:

```bash
python test_ai_integration.py
```

### Step 9: Deployment Considerations

#### Production Configuration

1. **Environment Variables**: Use secure environment variable management
2. **API Rate Limits**: Implement rate limiting for geocoding API calls
3. **Error Handling**: Add comprehensive error handling and logging
4. **Caching**: Implement Redis caching for improved performance
5. **Monitoring**: Add health checks and monitoring for AI services

#### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]
```

### Step 10: Security Best Practices

1. **API Key Management**: Store API keys securely, never in code
2. **Input Validation**: Always validate user inputs before processing
3. **Rate Limiting**: Implement rate limiting for AI API calls
4. **Error Handling**: Don't expose internal errors to users
5. **Data Sanitization**: Sanitize all user inputs
6. **HTTPS**: Always use HTTPS in production

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all AI files are in the correct directory structure
2. **API Key Issues**: Verify API keys are set correctly in environment variables
3. **Geocoding Failures**: Check internet connection and API availability
4. **Memory Issues**: Monitor memory usage for large-scale operations

### Debug Mode

Enable debug logging by setting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support

If you encounter any issues during integration:

1. Check the test script results
2. Verify all dependencies are installed correctly
3. Ensure API keys are properly configured
4. Check the application logs for error details
5. Verify network connectivity for external API calls

## 📈 Performance Optimization

1. **Caching**: Implement Redis for geocoding cache
2. **Async Processing**: Use async/await for better performance
3. **Batch Operations**: Process multiple requests in batches
4. **Database Indexing**: Add indexes for frequently queried fields
5. **Connection Pooling**: Use connection pooling for database connections

---

**Note**: This integration provides a complete AI-powered blood donation management system. Make sure to test thoroughly in a development environment before deploying to production.

For any questions or issues, please refer to the individual component documentation in the source files.

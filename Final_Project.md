# Final Project: Building AI Agents with LLMs

## Project Overview

This project aims to integrate your knowledge about AI agents and large language models (LLMs) into a practical application. You will develop a multi-agent system that can perform real-world tasks autonomously, simulating interactions between different agents that represent specialized roles.

## Learning Objectives

- Understand the architecture of multi-agent systems.
- Develop and deploy an application using Streamlit for user interaction.
- Utilize LLMs for intelligent decision-making and task execution.
- Implement asynchronous programming for interaction between agents.
- Explore Docker for containerization of your application.

## Project Requirements

- Basic knowledge of Python programming.
- Familiarity with APIs, especially OpenAI's API.
- Experience with Machine Learning libraries like NumPy and pandas.
- Installed libraries: Streamlit, OpenAI, pandas, requests.
- Optional: Basic knowledge of Docker for containerizing your application.

## Project Components

1. **Agent Design:**
   - Design at least three different agents:
     - **Weather Agent:** Responsible for predicting the best time to travel based on historical weather data.
     - **Hotel Agent:** Finds hotels matching user preferences.
     - **Itinerary Agent:** Creates a personalized travel itinerary.
    
2. **User Interface:**
   - Use Streamlit to build a web-based application where users can input their travel preferences.
   - Create input fields for user destination, preferences for hotels, and trip duration.

3. **Functionality:**
   - When the user submits their preferences, the application should:
     - Call the Weather Agent to get the optimal travel months.
     - Call the Hotel Agent to find suitable hotels.
     - Call the Itinerary Agent to generate a travel itinerary.
   - Display the results on the web interface, including the best months to visit, recommended hotels, and the itinerary.

4. **Asynchronous Programming:**
   - Implement asynchronous calls to ensure that the agents can operate independently without blocking each other.
   
5. **Deployment:**
   - Use Docker to containerize your application for deployment. This ensures that the application runs smoothly across different environments.

6. **Documentation:**
   - Prepare documentation detailing the choices made in the project, how to run the application, and any challenges faced during development.

## Deliverables

- A functional Streamlit application accessible via a web browser.
- Source code hosted on a public repository (e.g., GitHub).
- A Dockerfile for building your application containers.
- A documentation file (README.md) explaining the setup, usage, and functionality of your project.

## Evaluation Criteria

- **Functionality (40 points):** Application works as intended with all functionalities implemented.
- **Code Quality (20 points):** Code is well-organized, follows best practices, and is commented appropriately.
- **User Interface (20 points):** The Streamlit interface is user-friendly and aesthetically pleasing.
- **Documentation (20 points):** Comprehensive documentation is provided.

## Submission Deadline

- All projects must be submitted by [Insert Deadline Here].

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [OpenAI API Documentation](https://beta.openai.com/docs/)
- [Asynchronous Programming in Python](https://docs.python.org/3/library/asyncio.html)
- [Docker Documentation](https://docs.docker.com/get-started/)

Good luck, and have fun building your AI agents!
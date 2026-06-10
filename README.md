# End-to-End AI Support Ticket System

### Technical Work|AI Engineer| DOTMappers IT Pvt. Ltd.

A production-grade, zero-cost AI-powered analytics assistant designed to ingest customer support ticket data, automatically detect operational anomalies, and handle natural language questions using an advanced Text-to-Pandas LLM translation architecture.

---

## Architecture Overview

The system is built entirely in Python using a modular, three-layer data pipeline:

1. **Ingestion & Caching Layer (Pandas):** Loads the 500-row `support_tickets.csv` file into memory, automatically handling missing values for unresolved tickets (e.g., `resolution_time_hrs` and `customer_rating`) and parsing date parameters to establish a structured, high-speed data frame.
   
2. **Operational Anomaly Engine (Deterministic Python):**
   An automated background service that scans the live dataset using specific operational metrics to flag systemic risks:
   * **Backlog Staleness:** Flags open or escalated high/critical priority tickets that have remained unresolved for over 24 hours.
   * **Resolution Outliers:** Isolates historical tickets sitting in the top 5% (95th percentile) of resolution durations to pinpoint major processing bottlenecks.

3. **AI Natural Language Interface (Groq Cloud API):**
   Utilizes a deterministic, zero-temperature Text-to-Pandas compiler powered by the `llama-3.3-70b-versatile` model. Instead of relying on rigid hardcoded patterns or expensive, slow local infrastructure, it maps natural human questions directly into secure, highly optimized Python executable execution strings.

---

## Tech Stack & Tool Selection

* **Language:** Python only (Strict constraint)
* **Frontend UI:** Streamlit (Chosen for lightning-fast presentation, clear interactive navigation tabs, and transparent script-reloads)
* **Data Core:** Pandas (Provides high-performance vector alignment and filtering for analytical calculations)
* **LLM Engine:** Groq Cloud Service API (`llama-3.3-70b-versatile`) — Delivers enterprise-grade response latency at zero operational cost using open-source model footprints.
* **Environment Configuration:** Python-Dotenv (Protects system credentials from being exposed within raw source structures)

---

###Quick Start Setup Instructions
Follow these direct steps to launch the system locally. You can spin up this system at zero cost.

1. Clone & Navigate to the Project(on MAC/windows)
Bash/DOS

cd AI-Ticket-Analytics-System

2. Initialize and Activate the Virtual Environment

Bash/DOS 
python -m venv venv

# On Windows (Command Prompt) e.g C:\path\to your\AI-Ticket-Analytics-System>venv\Scripts\activate
venv\Scripts\activate


# On Mac/Linux
source venv/bin/activate

3. Install Pre-Compiled Dependencies


Bash
pip install --only-binary=:all: -r requirements.txt


4. Configure Your Secure Environment Key
Create a .env file in the root project directory and paste your free Groq API key:

Plaintext
GROQ_API_KEY=your_free_groq_api_key_here


5. Launch the Complete Application
Run the following single command to start the application server:

Bash
streamlit run app.py

The interface will automatically load in your browser at http://localhost:8501. 

📊 Sample Queries Handled Successfully
The system safely handles evaluation inquiries, automatically translating text to pandas expressions before showing results:

Query: "How many tickets are currently open?"

Backend Logic: df[df['status'] == 'Open'].shape[0]

Output: 111

Query: "What is the average customer rating for Technical category tickets?"

Backend Logic: df[df['category'] == 'Technical']['customer_rating'].mean()

Output: 3.74

Query: "Are there any anomalies in resolution times this week?"

Backend Logic: Uses specialized datetime isocalendar week extraction constraints to evaluate weekly anomalies safely.

Output: True

⚠️ Known Limitations & Production Scaling
In-Memory Constraints: The current setup processes calculations inside an in-memory Pandas DataFrame. For production systems scaling to millions of complaints, this layer would be migrated to an analytical SQL layer like PostgreSQL or DuckDB, with the LLM generating Text-to-SQL statements instead.

Security Guardrails: The system executes expressions using Python's native eval() function, which is ideal for a fast local assessment sandbox. In a multi-user enterprise production setting, strict code sanitization or specialized tool-calling validation abstractions would be implemented to mitigate remote code injection risks.




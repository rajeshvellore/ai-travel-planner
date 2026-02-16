import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="AI Travel Agent", page_icon="🌍", layout="wide")
st.title("🌍 Professional Multi-Agent Travel Planner")

with st.sidebar:
    st.header("🔑 API Setup")
    openai_key = st.text_input("OpenAI API Key", type="password")
    serper_key = st.text_input("Serper API Key", type="password")
    
    st.markdown("---")
    currency = st.selectbox("Currency", ["USD ($)", "INR (₹)"])
    unit = "$" if currency == "USD ($)" else "₹"

# --- 2. USER INPUTS (Now with 4 Columns for Duration) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    origin = st.text_input("Flying From", placeholder="e.g., Mumbai")
    city = st.text_input("Going To", placeholder="e.g., London")
with col2:
    month = st.text_input("Travel Month", placeholder="e.g., June 2026")
with col3:
    # --- NEW PARAMETER ---
    duration = st.number_input("Duration (Days)", min_value=1, max_value=30, value=3)
with col4:
    budget = st.number_input(f"Total Budget ({unit})", min_value=100, value=2000)

# --- 3. THE AGENTIC ENGINE ---
if st.button("Generate Complete Travel Plan"):
    if not openai_key or not serper_key or not origin or not city:
        st.error("Please fill in all inputs and API keys.")
    else:
        try:
            # Spinner now reflects the dynamic duration
            with st.spinner(f"Step 1: Researching {duration}-day trip details..."):
                os.environ["OPENAI_API_KEY"] = openai_key
                os.environ["SERPER_API_KEY"] = serper_key

                search_tool = SerperDevTool()
                llm = ChatOpenAI(model="gpt-4o-mini")

                # --- AGENTS (Goals now include duration) ---
                researcher = Agent(
                    role="Local Destination Expert",
                    goal=f"Discover activities and weather in {city} for {month}",
                    backstory="Local expert with deep knowledge of hidden gems.",
                    tools=[search_tool], llm=llm, verbose=True
                )

                transporter = Agent(
                    role="Global Transport Specialist",
                    goal=f"Find the most efficient transport from {origin} to {city}",
                    backstory="Master of flight routes and transit costs.",
                    tools=[search_tool], llm=llm, verbose=True
                )

                logistics_pro = Agent(
                    role="Travel Logistics Pro",
                    goal=f"Assemble a {duration}-day itinerary within budget",
                    backstory="Meticulous planner who balances all costs.",
                    tools=[search_tool], llm=llm, verbose=True
                )

                # --- PHASE 1: PRELIMINARY DATA ---
                research_task = Task(
                    description=f"Identify the best sights in {city} suitable for a {duration}-day trip in {month}.",
                    expected_output="A report on destination highlights.",
                    agent=researcher
                )

                transport_task = Task(
                    description=f"Find flight/train options from {origin} to {city}. Provide costs in {unit}.",
                    expected_output=f"A list of travel options with estimated prices in {unit}.",
                    agent=transporter
                )

                preliminary_crew = Crew(
                    agents=[researcher, transporter],
                    tasks=[research_task, transport_task],
                    process=Process.sequential
                )
                
                preliminary_results = preliminary_crew.kickoff()
                transport_info = str(preliminary_results)

                # --- UPDATED: PARAMETERIZED BUDGET VALIDATION ---
                # We calculate min stay cost based on duration
                daily_min = 100 if unit == "$" else 8000
                min_stay_required = daily_min * duration
                
                check_prompt = f"""
                Analyze the following travel data:
                User Budget: {unit}{budget}
                Duration: {duration} days
                Transport Options Found: {transport_info}
                
                Requirement: A {duration}-day stay in {city} typically requires a minimum of {unit}{min_stay_required} for hotel and food.
                
                If the cheapest transport + {unit}{min_stay_required} exceeds the User Budget, respond with:
                'INSUFFICIENT: [Estimated Minimum Total Amount Needed]'
                Otherwise, respond with: 'SUFFICIENT'
                """
                
                validation_check = llm.predict(check_prompt)

                if "INSUFFICIENT" in validation_check:
                    min_needed = validation_check.split(":")[1].strip()
                    st.error("❌ Budget Not Sufficient")
                    st.warning(f"For a {duration}-day trip to {city}, you need at least **{unit}{min_needed}** to cover transport and a decent stay.")
                    st.stop()

            # --- PHASE 2: FINAL PLANNING ---
            with st.spinner(f"Step 2: Budget is sufficient! Finalizing {duration}-day itinerary..."):
                itinerary_task = Task(
                    description=(
                        f"Create a detailed {duration}-day itinerary in {city}. Use research for activities and transport "
                        f"data. Total cost (Transport + Stay) must stay under {unit}{budget}."
                    ),
                    expected_output=f"A {duration}-day Markdown itinerary with prices in {unit}.",
                    agent=logistics_pro,
                    context=[research_task, transport_task]
                )

                final_crew = Crew(agents=[logistics_pro], tasks=[itinerary_task])
                final_plan = final_crew.kickoff()

                st.success(f"✅ Your {duration}-Day Plan is Ready!")
                st.markdown("---")
                
                if hasattr(final_plan, 'raw'):
                    st.markdown(final_plan.raw)
                else:
                    st.markdown(str(final_plan))

        except Exception as e:
            st.error(f"Something went wrong: {e}")
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

# --- 2. USER INPUTS ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    origin = st.text_input("Flying From", placeholder="e.g., Mumbai")
    city = st.text_input("Going To", placeholder="e.g., London")
with col2:
    month = st.text_input("Travel Month", placeholder="e.g., June 2026")
with col3:
    duration = st.number_input("Duration (Days)", min_value=1, max_value=30, value=3)
with col4:
    people = st.number_input("Number of People", min_value=1, max_value=20, value=1)
with col5:
    budget = st.number_input(f"Total Budget ({unit})", min_value=100, value=2000)

# --- 3. THE AGENTIC ENGINE ---
if st.button("Generate Complete Travel Plan"):
    if not openai_key or not serper_key or not origin or not city:
        st.error("Please fill in all inputs and API keys.")
    else:
        try:
            with st.spinner(f"Step 1: Researching trip for {people} person(s)..."):
                os.environ["OPENAI_API_KEY"] = openai_key
                os.environ["SERPER_API_KEY"] = serper_key

                search_tool = SerperDevTool()
                llm = ChatOpenAI(model="gpt-4o-mini")

                # --- AGENTS ---
                researcher = Agent(
                    role="Local Destination Expert",
                    goal=f"Discover activities in {city} for {month} suitable for {people} people",
                    backstory="Local expert focusing on group-friendly gems and weather.",
                    tools=[search_tool], llm=llm, verbose=True
                )

                transporter = Agent(
                    role="Global Transport Specialist",
                    goal=f"Find flight details from {origin} to {city} for {people} people",
                    backstory="Expert in group travel logistics and ticket costs.",
                    tools=[search_tool], llm=llm, verbose=True
                )

                logistics_pro = Agent(
                    role="Travel Logistics Pro",
                    goal=f"Assemble a {duration}-day itinerary for {people} people and create a cost breakdown",
                    backstory="Meticulous planner balancing flights and group costs with a final split-bill analysis.",
                    tools=[search_tool], llm=llm, verbose=True
                )

                # --- PHASE 1: PRELIMINARY DATA ---
                research_task = Task(
                    description=f"Identify top 5 sights in {city} for {people} people during {month}.",
                    expected_output="A report on destination highlights.",
                    agent=researcher
                )

                transport_task = Task(
                    description=(
                        f"Find flight options from {origin} to {city} for {people} travelers. "
                        f"Provide the TOTAL cost for all {people} people in {unit}.\n"
                        "- Include Airline, Flight Number, and Times."
                    ),
                    expected_output=f"A list containing flight details and total cost for {people} in {unit}.",
                    agent=transporter
                )

                preliminary_crew = Crew(
                    agents=[researcher, transporter],
                    tasks=[research_task, transport_task],
                    process=Process.sequential
                )
                
                preliminary_results = preliminary_crew.kickoff()
                transport_info = str(preliminary_results)

                # --- BUDGET VALIDATION ---
                daily_min_per_person = 80 if unit == "$" else 6000 
                total_stay_min = daily_min_per_person * duration * people
                
                check_prompt = f"""
                Analyze travel data:
                Budget: {unit}{budget}. People: {people}. Duration: {duration} days.
                Transport Found: {transport_info}
                Rule: A {duration}-day stay for {people} people typically requires {unit}{total_stay_min} for food/hotel.
                
                If (Transport + Stay) > {budget}, return 'INSUFFICIENT: [Estimated Min Total]'.
                Otherwise return 'SUFFICIENT'.
                """
                
                validation_check = llm.predict(check_prompt)

                if "INSUFFICIENT" in validation_check:
                    min_needed = validation_check.split(":")[1].strip()
                    st.error("❌ Budget Not Sufficient")
                    st.warning(f"For {people} people for {duration} days, you need at least **{unit}{min_needed}**.")
                    st.stop()

            # --- PHASE 2: FINAL PLANNING ---
            with st.spinner("Step 2: Budget sufficient! Finalizing itinerary and cost split..."):
                itinerary_task = Task(
                    description=(
                        f"Create a {duration}-day itinerary for {people} people in {city}. \n"
                        "MANDATORY:\n"
                        "1. Include 'Travel Logistics' at the top with total flight costs.\n"
                        "2. Provide the daily itinerary.\n"
                        "3. AT THE VERY END, provide a '💰 Group Cost Summary' table with: \n"
                        "- Total Flight Cost\n"
                        "- Estimated Total Hotel/Food Cost\n"
                        "- Total Trip Cost\n"
                        "- COST PER PERSON."
                    ),
                    expected_output=f"A {duration}-day Markdown plan for {people} people in {unit} including a cost-split table.",
                    agent=logistics_pro,
                    context=[research_task, transport_task]
                )

                final_crew = Crew(agents=[logistics_pro], tasks=[itinerary_task])
                final_plan = final_crew.kickoff()

                # --- FINAL DISPLAY ---
                st.success(f"✅ Your {duration}-Day Plan for {people} is Ready!")
                st.markdown("---")
                
                if hasattr(final_plan, 'raw'):
                    st.markdown(final_plan.raw)
                else:
                    st.markdown(str(final_plan))

        except Exception as e:
            st.error(f"Something went wrong: {e}")
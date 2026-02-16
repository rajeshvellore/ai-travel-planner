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
col1, col2, col3 = st.columns(3)
with col1:
    origin = st.text_input("Flying From", placeholder="e.g., Mumbai")
    city = st.text_input("Going To", placeholder="e.g., London")
with col2:
    month = st.text_input("Travel Month", placeholder="e.g., June 2026")
with col3:
    budget = st.number_input(f"Total Budget ({unit})", min_value=100, value=2000)
# --- 3. THE AGENTIC ENGINE ---
if st.button("Generate Complete Travel Plan"):
    if not openai_key or not serper_key or not origin or not city:
        st.error("Please fill in all inputs and API keys.")
    else:
        try:
            with st.spinner("Our AI team is collaborating on your trip..."):
                os.environ["OPENAI_API_KEY"] = openai_key
                os.environ["SERPER_API_KEY"] = serper_key

                search_tool = SerperDevTool()
                llm = ChatOpenAI(model="gpt-4o-mini")

                # AGENT 1: The Researcher (Session: Role-based behavior)
                researcher = Agent(
                    role="Local Destination Expert",
                    goal=f"Discover the best activities, weather, and hidden gems in {city} for {month}",
                    backstory="You are a local expert with deep knowledge of {city}. You know the best times to visit and local secrets.",
                    tools=[search_tool],
                    llm=llm,
                    verbose=True
                )

                # AGENT 2: The Transport Specialist (New Agent)
                transporter = Agent(
                    role="Global Transport Specialist",
                    goal=f"Find the most efficient and cost-effective transport from {origin} to {city}",
                    backstory="You are a master of flight routes, train schedules, and transit costs. You focus on logistics.",
                    tools=[search_tool],
                    llm=llm,
                    verbose=True
                )

                # AGENT 3: The Logistics Pro (The Assembler)
                logistics_pro = Agent(
                    role="Travel Logistics Pro",
                    goal=f"Combine transport and research into a final itinerary within a {unit}{budget} budget",
                    backstory="You are a meticulous travel agent. You balance the travel costs with hotel and activity costs.",
                    tools=[search_tool],
                    llm=llm,
                    verbose=True
                )

                # --- DEFINING TASKS ---
                research_task = Task(
                    description=f"Identify top 5 sights and current weather in {city} during {month}.",
                    expected_output="A report on destination highlights and local conditions.",
                    agent=researcher
                )

                transport_task = Task(
                    description=f"Find flight/train options from {origin} to {city}. Provide costs in {unit}.",
                    expected_output=f"A list of travel options with estimated prices in {unit}.",
                    agent=transporter
                )

                itinerary_task = Task(
                    description=(
                        f"Create a 3-day itinerary in {city}. Use the research for activities and the transport "
                        f"data for travel. The total cost (Transport + Stay) must not exceed {unit}{budget}."
                    ),
                    expected_output=f"A final Markdown itinerary including Transport details and Daily breakdown in {unit}.",
                    agent=logistics_pro,
                    context=[research_task, transport_task] # RAG-like context sharing between agents
                )

                # --- ORCHESTRATION ---
                crew = Crew(
                    agents=[researcher, transporter, logistics_pro],
                    tasks=[research_task, transport_task, itinerary_task],
                    process=Process.sequential,
                    verbose=True
                )

                final_plan = crew.kickoff()

                # --- DISPLAY ---
                st.success("✅ Your Plan is Ready!")
                st.markdown("---")
                # st.markdown(final_plan.raw)
                # Smart Display: Check if it's an object with .raw or just a string
                if hasattr(final_plan, 'raw'):
                    st.markdown(final_plan.raw)
                else:
                    st.markdown(str(final_plan)) # If it's already a string, just show it


        except Exception as e:
            st.error(f"Something went wrong: {e}")
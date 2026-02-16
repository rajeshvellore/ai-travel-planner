import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool

# --- UI CONFIGURATION ---
st.set_page_config(page_title="AI Travel Pro", page_icon="🌍", layout="wide")

st.title("🌍 Agentic Travel Planner")
st.markdown("This system uses a **Multi-Agent** approach to research and plan your trip.")

# --- SIDEBAR: CREDENTIALS ---
with st.sidebar:
    st.header("🔑 API Configuration")
    os.environ["OPENAI_API_KEY"] = st.text_input("OpenAI API Key", type="password")
    os.environ["SERPER_API_KEY"] = st.text_input("Serper API Key", type="password")
    st.info("Get an OpenAI key at [platform.openai.com](https://platform.openai.com) and a Serper key at [serper.dev](https://serper.dev)")

# --- USER INPUTS ---
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.text_input("Destination City", placeholder="e.g. Rome")
    with col2:
        month = st.text_input("Travel Month", placeholder="e.g. October 2026")
    with col3:
        budget = st.number_input("Budget ($)", min_value=100, value=2000)

# --- THE AGENTIC ENGINE ---
if st.button("Generate My Itinerary"):
    if not os.environ["OPENAI_API_KEY"] or not os.environ["SERPER_API_KEY"]:
        st.error("Please provide both API keys in the sidebar to proceed.")
    else:
        try:
            with st.spinner("🕵️ Agents are researching flights, hotels, and sights..."):
                
                # 1. Initialize Tools & LLM
                search_tool = SerperDevTool()
                llm = ChatOpenAI(model="gpt-4o-mini")

                # 2. Define specialized Agents (Session: Multi-Agent Systems)
                researcher = Agent(
                    role="Local Destination Expert",
                    goal=f"Find the best attractions and weather for {city} in {month}",
                    backstory="You are a local guide who knows every hidden gem in the city.",
                    tools=[search_tool],
                    llm=llm,
                    verbose=True
                )

                logistics_expert = Agent(
                    role="Travel Logistics Pro",
                    goal=f"Find top-rated hotels and transit options in {city} within a total budget of ${budget}",
                    backstory="You are an expert at finding the best value-for-money luxury stays and transport.",
                    tools=[search_tool],
                    llm=llm,
                    verbose=True
                )

                # 3. Define Tasks (Session: Prompt Engineering)
                research_task = Task(
                    description=f"Identify 5 must-visit places in {city} and 3 local food specialties. Check weather for {month}.",
                    expected_output="A summary report on sights, food, and weather.",
                    agent=researcher
                )

                logistics_task = Task(
                    description=f"Suggest 3 hotels and best ways to get around {city} for 5 days. Total trip stay must fit budget: ${budget}.",
                    expected_output="A list of hotels with prices and transport tips.",
                    agent=logistics_expert
                )

                # 4. Assemble the Crew (Session: Orchestration)
                crew = Crew(
                    agents=[researcher, logistics_expert],
                    tasks=[research_task, logistics_task],
                    process=Process.sequential # First research, then logistics
                )

                final_itinerary = crew.kickoff()

                # --- DISPLAY RESULTS ---
                st.success("✅ Your Plan is Ready!")
                st.markdown("---")
                st.subheader(f"Trip to {city}")
                st.markdown(final_itinerary)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
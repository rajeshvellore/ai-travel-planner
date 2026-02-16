import streamlit as st
import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="AI Travel Agent", page_icon="✈️", layout="centered")
st.title("✈️ AI Agentic Travel Planner")

# --- 2. CURRENCY & EXCHANGE RATE LOGIC ---
# Using a fixed rate for stability (1 USD = 83 INR). 
# In 2026, you could also use an API to fetch this live.
EXCHANGE_RATE = 83.0 

with st.sidebar:
    st.header("Setup")
    openai_key = st.text_input("OpenAI API Key", type="password")
    serper_key = st.text_input("Serper API Key", type="password")
    
    st.markdown("---")
    currency = st.selectbox("Select Currency", ["USD ($)", "INR (₹)"])
    
    # Define symbols and budget limits based on currency
    if currency == "USD ($)":
        unit = "$"
        min_val, max_val, default_val = 500, 10000, 2000
    else:
        unit = "₹"
        # Convert the USD limits to INR for the slider
        min_val, max_val, default_val = 40000, 800000, 150000

# --- 3. USER INPUT FIELDS ---
col1, col2 = st.columns(2)
with col1:
    city = st.text_input("Where to?", placeholder="e.g. Tokyo")
with col2:
    month = st.text_input("When?", placeholder="e.g. April 2026")

budget = st.slider(f"Total Budget ({unit})", min_val, max_val, default_val)

# --- 4. THE AI LOGIC ---
if st.button("Plan My Trip"):
    if not openai_key or not serper_key:
        st.error("Please enter API keys in the sidebar!")
    else:
        try:
            with st.spinner("Agents are researching..."):
                os.environ["OPENAI_API_KEY"] = openai_key
                os.environ["SERPER_API_KEY"] = serper_key

                # Normalize budget to USD for the AI's "global" understanding
                # But tell it to respond in the user's chosen currency
                budget_in_usd = budget if unit == "$" else budget / EXCHANGE_RATE

                search_tool = SerperDevTool()
                llm = ChatOpenAI(model="gpt-4o-mini")

                planner = Agent(
                    role="Expert Travel Concierge",
                    goal=f"Create a high-value itinerary for {city}",
                    backstory="You are an expert travel agent. You always provide costs in the user's local currency.",
                    tools=[search_tool],
                    llm=llm
                )

                task = Task(
                    description=(
                        f"Plan a 3-day trip to {city} in {month}. "
                        f"The total budget is {unit}{budget}. "
                        f"Ensure all cost estimates in the itinerary are shown in {currency}."
                    ),
                    expected_output=f"A structured 3-day itinerary with prices listed in {unit}.",
                    agent=planner
                )

                crew = Crew(agents=[planner], tasks=[task])
                result = crew.kickoff()

                st.success("Trip Planned!")
                st.markdown(f"### Your Custom Itinerary ({unit})")
                st.write(result.raw)

        except Exception as e:
            st.error(f"Error: {e}")
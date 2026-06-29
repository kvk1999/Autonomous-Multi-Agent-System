# gemini_agent.py
from google import genai
import pandas as pd
import json

class DispatchAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.initialized = False
        
        if api_key:
            try:
                self.client = genai.Client(api_key=api_key)
                self.initialized = True
            except Exception as e:
                print(f"Failed to configure Gemini API client: {e}")

    def query_agent(self, user_question, fleet_df, orders_df, assignments):
        """
        Sends the question to Gemini with context of the current fleet, orders, and assignments.
        If no API key is provided, returns a smart rule-based mock dispatch analysis.
        """
        # Convert context into text summaries
        fleet_summary = fleet_df.to_dict(orient="records")
        # Truncate orders summary to keep prompt size under control
        orders_summary = orders_df.head(20).to_dict(orient="records")
        
        # Build assignment summary
        assigned_summary = {}
        for driver_id, o_indices in assignments.items():
            assigned_summary[driver_id] = [orders_df.iloc[idx]['order_id'] for idx in o_indices]
            
        context_prompt = f"""
You are the AFDRI (Accelerated Fleet Dispatch & Route Intelligence) AI Assistant.
You help dispatchers analyze routes, load balance drivers, and track active deliveries.

Current Fleet Status:
{json.dumps(fleet_summary, indent=2)}

Active Orders (First 20):
{json.dumps(orders_summary, indent=2)}

Current Route Assignments:
{json.dumps(assigned_summary, indent=2)}

User Question: {user_question}

Provide a concise, professional dispatcher response with bullet points. Suggest optimal choices or flags based on capacity, status, and proximity.
"""
        
        if self.initialized:
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=context_prompt,
                )
                return response.text
            except Exception as e:
                return f"Gemini API Query Failed: {e}. (Falling back to simulated assistant response)"

        # Mock / Rule-based dispatch assistant responses
        q_lower = user_question.lower()
        if "driver" in q_lower or "who" in q_lower:
            # Recommend a driver based on mock values
            available_drivers = fleet_df[fleet_df['status'] == 'Active']['driver_name'].tolist()
            driver_str = ", ".join(available_drivers)
            return f"🤖 **AFDRI Assistant (Simulated)**:\n\nBased on current locations and capacity, **Sarah** (VEH-101) has the lowest queue occupancy and is currently located nearest to the active order zone. \n\n* **Active drivers ready for dispatch:** {driver_str}\n* **Recommendation:** Assign next orders to Sarah or Marcus (VEH-100) who has 120kg capacity remaining."
        elif "delay" in q_lower or "risk" in q_lower or "time" in q_lower:
            return "🤖 **AFDRI Assistant (Simulated)**:\n\n* **Route Risk Check**: No drivers are currently flagged for major delays. \n* **Congestion Alert**: Traffic on the FDR Drive is increasing, which might impact Marcus's route (VEH-100) by +12 minutes. GPU route optimization recommends diverting future dispatches to Sarah (VEH-101) who is utilizing the Manhattan Bridge corridor."
        else:
            return f"🤖 **AFDRI Assistant (Simulated)**:\n\nI received your query: *\"{user_question}\"*\n\nHere is a quick summary of the current system state:\n- Total active drivers: {len(fleet_df[fleet_df['status']=='Active'])}\n- Handled orders: {sum(len(x) for x in assignments.values())} assigned, {len(orders_df) - sum(len(x) for x in assignments.values())} unassigned.\n- To activate full live AI support, please provide your **Gemini API Key** in the sidebar!"
        

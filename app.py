
import streamlit as st
import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import uuid


# --- Page Configuration ---
st.set_page_config(
    page_title="AI Health Agent & Emergency Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (from dashboard)
st.markdown("""
<style>
    .alert-critical {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        animation: pulse 2s infinite;
    }
    .alert-high {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .alert-moderate {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(220, 53, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-connected {
        color: #28a745;
    }
    .status-error {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# --- EmergencyDashboard class from user code ---
API_BASE_URL = "http://127.0.0.1:8000"
REFRESH_INTERVAL = 10  # seconds

class EmergencyDashboard:
    def __init__(self):
        self.api_connected = False
        self.last_check = None
    def check_api_connection(self) -> bool:
        try:
            # Use the lightweight /status endpoint (quick)
            response = requests.get(f"{API_BASE_URL}/status", timeout=5)
            if response.status_code == 200:
                self.api_connected = True
                return True
        except Exception as e:
            st.error(f"API Connection Error: {e}")
        self.api_connected = False
        return False
    def get_current_alerts(self) -> List[Dict[str, Any]]:
        try:
            response = requests.get(f"{API_BASE_URL}/alerts/current", timeout=15)
            if response.status_code == 200:
                data = response.json()
                return data.get('alerts', [])
            else:
                st.error(f"API Error: {response.status_code}")
                return []
        except Exception as e:
            st.error(f"Error fetching alerts: {e}")
            return []
    def get_simulation_status(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{API_BASE_URL}/simulation/status", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {"available": False, "error": "API unavailable"}
        except Exception as e:
            return {"available": False, "error": str(e)}
    def run_simulation(self, simulation_type: str) -> Dict[str, Any]:
        try:
            response = requests.post(f"{API_BASE_URL}/simulation/run/{simulation_type}", timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    def reset_simulation_data(self) -> bool:
        try:
            response = requests.post(f"{API_BASE_URL}/simulation/reset", timeout=10)
            return response.status_code == 200
        except Exception as e:
            st.error(f"Error resetting data: {e}")
            return False
    def process_emergency_report(self, query: str) -> Dict[str, Any]:
        try:
            data = {
                "query": query,
                "session_id": f"streamlit-{int(time.time())}"
            }
            response = requests.post(f"{API_BASE_URL}/emergency/process", json=data, timeout=20)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    def display_alert(self, alert: Dict[str, Any]):
        # Defensive: handle if alert is a string or dict
        if isinstance(alert, str):
            st.warning(f"Alert: {alert}")
            return
        raw_alert = alert.get('raw_data', alert)
        if isinstance(raw_alert, str):
            try:
                raw_alert = json.loads(raw_alert)
            except Exception:
                st.warning(f"Alert: {raw_alert}")
                return
        alert_level = raw_alert.get('alert_level', 'UNKNOWN').lower()
        css_class = f"alert-{alert_level}" if alert_level in ['critical', 'high', 'moderate'] else "alert-moderate"
        title = raw_alert.get('title', 'Emergency Alert')
        summary = raw_alert.get('summary', 'Alert details unavailable')
        alert_type = raw_alert.get('alert_type', 'unknown')
        timestamp = raw_alert.get('timestamp', datetime.now().isoformat())
        try:
            alert_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = alert_time.strftime('%Y-%m-%d %H:%M:%S')
        except:
            time_str = timestamp
        alert_html = f"""
        <div class="{css_class}">
            <h3>🚨 {title}</h3>
            <p><strong>Level:</strong> {alert_level.upper()}</p>
            <p><strong>Type:</strong> {alert_type.replace('_', ' ').title()}</p>
            <p><strong>Summary:</strong> {summary}</p>
            <p><strong>Time:</strong> {time_str}</p>
        </div>
        """
        st.markdown(alert_html, unsafe_allow_html=True)
        details = raw_alert.get('details', {})
        if details:
            with st.expander("View Detailed Information"):
                st.json(details)
        recommendations = raw_alert.get('recommendations', [])
        immediate_actions = raw_alert.get('immediate_actions', [])
        if recommendations:
            st.write("**Recommendations:**")
            for rec in recommendations:
                st.write(f"• {rec}")
        if immediate_actions:
            st.write("**Immediate Actions:**")
            for action in immediate_actions:
                st.write(f"• {action}")
        contacts = raw_alert.get('emergency_contacts', {})
        if contacts:
            st.write("**Emergency Contacts:**")
            for service, number in contacts.items():
                st.write(f"• {service.replace('_', ' ').title()}: {number}")


# --- Main App Tabs ---
tabs = st.tabs(["💬 Health Agent Chat", "🚨 Emergency Dashboard", "📡 Gov Alerts & SMS"])

# --- Tab 1: Health Agent Chat (original chat UI) ---
with tabs[0]:
    st.title("AI Health Agent")
    st.caption(f"Operating for Warangal, Telangana | {datetime.now().strftime('%B %d, %Y')}")

    # --- Check and Display Emergency Status Banner ---
    try:
        status_response = requests.get(f"{API_BASE_URL}/status")
        if status_response.status_code == 200 and status_response.json().get("emergency_mode_active"):
            st.error(
                " **EMERGENCY MODE ACTIVE** | The system is currently operating in emergency response mode. Please describe your situation or symptoms for immediate classification.",
            )
            default_message = "EMERGENCY ACTIVE: Please describe your situation or symptoms."
        else:
            default_message = "Hello! How can I help you with your health today?"
    except requests.exceptions.ConnectionError:
        st.warning("Could not connect to the backend server.")
        default_message = "Connection to the server failed. Please ensure the backend is running."

    # --- Session State Initialization ---
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": default_message}]

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Handle User Input ---
    if prompt := st.chat_input("Ask about health or describe your symptoms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            try:
                api_url = f"{API_BASE_URL}/chat"
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id
                }
                response = requests.post(api_url, json=payload)
                response.raise_for_status()
                agent_response = response.json()["response"]
                message_placeholder.markdown(agent_response)
                st.session_state.messages.append({"role": "assistant", "content": agent_response})
            except requests.exceptions.RequestException as e:
                error_message = f"Could not connect to the AI backend. Please make sure the FastAPI server is running. Error: {e}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# --- Tab 2: Emergency Dashboard (user's dashboard UI) ---
with tabs[1]:
    dashboard = EmergencyDashboard()
    st.title("🚨 Emergency Alert System - Warangal District")
    st.markdown("Real-time Emergency Monitoring & Response System")

    st.sidebar.header("Control Panel")
    api_status = dashboard.check_api_connection()
    if api_status:
        st.sidebar.markdown('<p class="status-connected">🟢 API Connected</p>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<p class="status-error">🔴 API Disconnected</p>', unsafe_allow_html=True)
    auto_refresh = st.sidebar.checkbox("Auto-refresh alerts", value=True)
    refresh_rate = st.sidebar.slider("Refresh interval (seconds)", 5, 60, REFRESH_INTERVAL)
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    st.sidebar.subheader("Simulation Controls")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🦠 Simulate Outbreak"):
            with st.spinner("Running outbreak simulation..."):
                result = dashboard.run_simulation("outbreak")
                if 'error' not in result:
                    st.success(f"Simulation completed: {result.get('reports_processed', 0)} reports")
                    if result.get('alerts_generated', 0) > 0:
                        st.success("🚨 Alert generated!")
                else:
                    st.error(f"Simulation failed: {result['error']}")
    with col2:
        if st.button("🌊 Simulate Disaster"):
            with st.spinner("Running disaster simulation..."):
                result = dashboard.run_simulation("disaster")
                if 'error' not in result:
                    st.success(f"Simulation completed: {result.get('reports_processed', 0)} reports")
                    if result.get('alerts_generated', 0) > 0:
                        st.success("🚨 Alert generated!")
                else:
                    st.error(f"Simulation failed: {result['error']}")
    if st.sidebar.button("🗑️ Reset All Data"):
        if dashboard.reset_simulation_data():
            st.success("All simulation data reset successfully")
        else:
            st.error("Failed to reset simulation data")

    if not api_status:
        st.error("Cannot connect to the emergency system API. Please ensure the API is running on localhost:8000")
        st.code("python main.py", language="bash")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["🚨 Active Alerts", "📊 System Status", "📝 Report Emergency", "📈 Analytics"])
        with tab1:
            st.header("Active Emergency Alerts")
            alerts = dashboard.get_current_alerts()
            if alerts:
                st.warning(f"⚠️ {len(alerts)} active emergency alert(s)")
                for i, alert in enumerate(alerts):
                    st.subheader(f"Alert {i+1}")
                    dashboard.display_alert(alert)
                    st.divider()
            else:
                st.success("✅ No active alerts at this time")
                st.info("The system is continuously monitoring for emergency situations...")
        with tab2:
            st.header("System Status")
            status = dashboard.get_simulation_status()
            if status.get('available'):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Reports", status.get('total_emergency_reports', 0))
                with col2:
                    st.metric("Outbreak Reports", status.get('total_epidemic_reports', 0))
                with col3:
                    st.metric("Active Alerts", status.get('current_alerts', 0))
                with col4:
                    st.metric("System Status", "Online" if api_status else "Offline")
                st.subheader("System Health")
                st.success("Emergency monitoring system operational")
                st.info(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.error(f"System Status Error: {status.get('error', 'Unknown error')}")
        with tab3:
            st.header("Report Emergency")
            with st.form("emergency_report_form"):
                st.write("Use this form to report an emergency situation:")
                emergency_type = st.selectbox(
                    "Emergency Type",
                    ["Disease Outbreak", "Natural Disaster", "Medical Emergency"]
                )
                description = st.text_area(
                    "Description",
                    placeholder="Describe the emergency situation, symptoms, or conditions...",
                    height=100
                )
                location = st.text_input("Location", value="Warangal, Telangana", disabled=True)
                submitted = st.form_submit_button("🚨 Submit Emergency Report")
                if submitted and description:
                    with st.spinner("Processing emergency report..."):
                        result = dashboard.process_emergency_report(description)
                        if 'error' not in result:
                            st.success("Emergency report submitted successfully!")
                            st.subheader("Emergency Response")
                            st.write(result.get('emergency_response', 'Report processed'))
                            if result.get('public_alert_generated'):
                                st.warning("🚨 This report triggered a public alert!")
                                st.json(result['public_alert_generated'])
                        else:
                            st.error(f"Error processing report: {result['error']}")
                elif submitted:
                    st.error("Please provide a description of the emergency")
        with tab4:
            st.header("Analytics Dashboard")
            st.info("Analytics dashboard coming soon...")
            st.write("This section will include:")
            st.write("• Alert frequency trends")
            st.write("• Geographic distribution of reports")
            st.write("• Response time metrics")
            st.write("• System performance indicators")
        if auto_refresh and api_status:
            time.sleep(refresh_rate)
            st.rerun()

# ---------------------------------------------------------------------------
# Tab 3: Government Alerts & SMS
# ---------------------------------------------------------------------------
with tabs[2]:
    st.title("📡 Government Outbreak Alerts & SMS")
    st.markdown("Monitor official WHO/NDMA feeds and manage emergency SMS subscribers.")

    gov_tab1, gov_tab2, gov_tab3 = st.tabs([
        "🌐 Government Alerts",
        "📱 SMS Subscribers",
        "✉️ Send Manual SMS",
    ])

    # -----------------------------------------------------------------------
    # Sub-tab 1: Government Alerts
    # -----------------------------------------------------------------------
    with gov_tab1:
        st.header("Government & International Outbreak Alerts")
        st.markdown("""
        Alerts are fetched automatically every **30 minutes** from:
        - 🌍 **WHO Disease Outbreak News** — official WHO RSS
        - 🚨 **NDMA India** — National Disaster Management Authority RSS
        - 📡 **ProMED** — International Society for Infectious Diseases

        Filtered for India/Telangana relevance.
        """)

        col_fetch, col_cache = st.columns([1, 2])
        with col_fetch:
            if st.button("🔄 Fetch Now", key="fetch_gov_alerts"):
                with st.spinner("Fetching government alert feeds..."):
                    try:
                        resp = requests.post(f"{API_BASE_URL}/gov-alerts/fetch", timeout=30)
                        if resp.status_code == 200:
                            st.success(f"✅ Fetched {resp.json().get('count', 0)} alert(s)")
                        else:
                            st.error(f"Error {resp.status_code}: {resp.text[:200]}")
                    except Exception as e:
                        st.error(f"Could not reach API: {e}")

        # Fetch cached alerts
        try:
            cached_resp = requests.get(f"{API_BASE_URL}/gov-alerts/cached", timeout=10)
            if cached_resp.status_code == 200:
                cached_data = cached_resp.json()
                gov_alerts = cached_data.get("alerts", [])
                last_fetched = cached_data.get("last_fetched", "Not yet")

                with col_cache:
                    st.caption(f"🕐 Last fetched: {last_fetched}  |  {len(gov_alerts)} alert(s) cached")

                if gov_alerts:
                    # Group by source
                    by_source: dict = {}
                    for a in gov_alerts:
                        src = a.get("source", "Unknown")
                        by_source.setdefault(src, []).append(a)

                    for source_name, source_alerts in by_source.items():
                        st.subheader(f"📌 {source_name} ({len(source_alerts)} alerts)")
                        for alert in source_alerts:
                            level = alert.get("alert_level", "LOW")
                            level_colors = {
                                "CRITICAL": "#f8d7da",
                                "HIGH": "#fff3cd",
                                "MODERATE": "#d4edda",
                                "LOW": "#e2e3e5",
                            }
                            bg = level_colors.get(level, "#e2e3e5")
                            title = alert.get("title", "Alert")
                            summary = alert.get("summary", "")
                            link = alert.get("link", "#")
                            ts = alert.get("timestamp", "")

                            st.markdown(
                                f"""
                                <div style="background:{bg}; border-radius:8px; padding:14px; margin:8px 0;">
                                    <strong>[{level}] {title}</strong><br/>
                                    <small>{ts[:10] if ts else ''}</small><br/>
                                    {summary}<br/>
                                    <a href="{link}" target="_blank">🔗 Read more</a>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("No government alerts cached yet. Click 'Fetch Now' or wait for the automatic 30-minute poll.")
            else:
                st.error(f"Could not fetch cached alerts: {cached_resp.status_code}")
        except Exception as e:
            st.warning(f"API not reachable: {e}")

    # -----------------------------------------------------------------------
    # Sub-tab 2: SMS Subscribers
    # -----------------------------------------------------------------------
    with gov_tab2:
        st.header("SMS Alert Subscribers")
        st.markdown("""
        Subscribers receive an automatic SMS whenever a public emergency alert is generated.
        Admin numbers from `.env → SMS_ALERT_SUBSCRIBERS` are seeded at startup.
        """)

        # List current subscribers
        try:
            sub_resp = requests.get(f"{API_BASE_URL}/sms/subscribers", timeout=10)
            if sub_resp.status_code == 200:
                sub_data = sub_resp.json()
                total_active = sub_data.get("total_active", 0)
                all_subs = sub_data.get("subscribers", [])

                st.metric("Active Subscribers", total_active)

                if all_subs:
                    sub_rows = []
                    for s in all_subs:
                        sub_rows.append({
                            "Phone": s["phone_number"],
                            "Name": s["name"],
                            "Status": "✅ Active" if s["active"] else "❌ Inactive",
                            "Subscribed At": s["subscribed_at"][:19] if s["subscribed_at"] else "",
                        })
                    st.dataframe(pd.DataFrame(sub_rows), use_container_width=True)
                else:
                    st.info("No subscribers yet.")
            else:
                st.error(f"Could not fetch subscribers: {sub_resp.status_code}")
        except Exception as e:
            st.warning(f"API not reachable: {e}")

        st.divider()
        col_add, col_remove = st.columns(2)

        with col_add:
            st.subheader("➕ Add Subscriber")
            with st.form("add_subscriber_form"):
                new_phone = st.text_input(
                    "Phone Number (E.164)",
                    placeholder="+919148033536",
                    key="new_sub_phone",
                )
                new_name = st.text_input("Name", value="Subscriber", key="new_sub_name")
                add_submitted = st.form_submit_button("Add Subscriber")
                if add_submitted and new_phone:
                    try:
                        r = requests.post(
                            f"{API_BASE_URL}/sms/subscribe",
                            json={"phone_number": new_phone, "name": new_name},
                            timeout=10,
                        )
                        if r.status_code == 200:
                            st.success(f"✅ Subscribed {new_phone}")
                            st.rerun()
                        else:
                            st.error(f"Error: {r.text[:200]}")
                    except Exception as e:
                        st.error(f"API error: {e}")
                elif add_submitted:
                    st.warning("Please enter a phone number.")

        with col_remove:
            st.subheader("➖ Remove Subscriber")
            with st.form("remove_subscriber_form"):
                rm_phone = st.text_input(
                    "Phone Number (E.164)",
                    placeholder="+919148033536",
                    key="rm_sub_phone",
                )
                rm_submitted = st.form_submit_button("Remove Subscriber")
                if rm_submitted and rm_phone:
                    try:
                        r = requests.delete(
                            f"{API_BASE_URL}/sms/unsubscribe",
                            params={"phone_number": rm_phone},
                            timeout=10,
                        )
                        if r.status_code == 200:
                            st.success(f"✅ Removed {rm_phone}")
                            st.rerun()
                        elif r.status_code == 404:
                            st.warning("Number not found in subscriber list.")
                        else:
                            st.error(f"Error: {r.text[:200]}")
                    except Exception as e:
                        st.error(f"API error: {e}")
                elif rm_submitted:
                    st.warning("Please enter a phone number.")

    # -----------------------------------------------------------------------
    # Sub-tab 3: Manual SMS
    # -----------------------------------------------------------------------
    with gov_tab3:
        st.header("✉️ Send Manual SMS")
        st.markdown("Send a test or manual SMS to any number via Twilio.")

        with st.form("manual_sms_form"):
            sms_to = st.text_input(
                "Recipient Phone (E.164)",
                placeholder="+919148033536",
                key="manual_sms_to",
            )
            sms_body = st.text_area(
                "Message",
                placeholder="Type your message here...",
                height=120,
                max_chars=1600,
                key="manual_sms_body",
            )
            char_count = len(sms_body) if sms_body else 0
            st.caption(f"{char_count}/1600 characters")

            sms_submitted = st.form_submit_button("📤 Send SMS")
            if sms_submitted and sms_to and sms_body:
                with st.spinner("Sending SMS..."):
                    try:
                        r = requests.post(
                            f"{API_BASE_URL}/sms/send",
                            json={"to_number": sms_to, "message": sms_body},
                            timeout=20,
                        )
                        if r.status_code == 200:
                            result = r.json()
                            st.success(f"✅ SMS sent! SID: {result.get('sid', 'N/A')}")
                        else:
                            err = r.json().get("detail", r.text[:200])
                            st.error(f"❌ Send failed: {err}")
                    except Exception as e:
                        st.error(f"API error: {e}")
            elif sms_submitted:
                st.warning("Please fill in both recipient number and message.")
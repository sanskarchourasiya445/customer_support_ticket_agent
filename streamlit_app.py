import os
import uuid

import httpx
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Customer Support", page_icon="🎧")
st.title("Customer Support")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_assistant_extra(msg: dict) -> None:
    sources = msg.get("sources") or []
    ticket_id = msg.get("ticket_id")

    if sources:
        st.caption(f"📚 **Sources:** {', '.join(sources)}")
    if ticket_id:
        st.success(f"🎫 **Ticket Created:** `{ticket_id}`")


with st.sidebar:
    st.subheader("Session Info")
    st.text(f"Session ID:\n{st.session_state.session_id}")
    if st.button("New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_assistant_extra(message)

if prompt := st.chat_input("How can we help?"):
    clean_prompt = prompt.strip()
    if clean_prompt:
        st.session_state.messages.append({"role": "user", "content": clean_prompt})
        with st.chat_message("user"):
            st.write(clean_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Support agent is thinking..."):
                payload = {
                    "session_id": st.session_state.session_id,
                    "message": clean_prompt,
                }
                try:
                    with httpx.Client(timeout=30.0) as client:
                        response = client.post(f"{API_BASE_URL}/chat", json=payload)
                        if response.status_code == 200:
                            data = response.json()
                            resp_text = data.get("response", "")
                            sources = data.get("sources") or []
                            ticket_id = data.get("ticket_id")

                            st.write(resp_text)
                            render_assistant_extra({"sources": sources, "ticket_id": ticket_id})

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": resp_text,
                                "sources": sources,
                                "ticket_id": ticket_id,
                            })
                        elif response.status_code == 422:
                            st.error("Validation Error: Please provide a valid, non-empty message.")
                        elif response.status_code == 503:
                            st.error("Service Unavailable: The support agent is initializing or unavailable.")
                        elif response.status_code == 502:
                            detail = response.json().get("detail", "Error processing support request.")
                            st.error(f"Agent Processing Error: {detail}")
                        else:
                            st.error(f"Error {response.status_code}: {response.text}")
                except httpx.ConnectError:
                    st.error(f"Cannot connect to the backend server at `{API_BASE_URL}`. Please ensure the API is running.")
                except httpx.TimeoutException:
                    st.error("Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

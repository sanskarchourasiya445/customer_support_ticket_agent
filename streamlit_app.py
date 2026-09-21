import hashlib
import os
import uuid

import httpx
import streamlit as st


try:
    from src.config import load_settings
    _settings = load_settings()
    API_BASE_URL = os.getenv("API_BASE_URL", f"http://{_settings.api_host}:{_settings.api_port}")
except Exception:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Customer Support", page_icon="🎧")
st.title("Customer Support")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""
if "last_transcribed_audio_id" not in st.session_state:
    st.session_state.last_transcribed_audio_id = None




def render_assistant_extra(msg: dict) -> None:
    sources = msg.get("sources") or []
    ticket_id = msg.get("ticket_id")

    if sources:
        st.caption(f"📚 **Sources:** {', '.join(sources)}")
    if ticket_id:
        st.success(f"🎫 **Ticket Created:** `{ticket_id}`")


def process_user_query(query_text: str) -> None:
    clean_prompt = query_text.strip()
    if not clean_prompt:
        return

    st.session_state.messages.append({
        "role": "user",
        "content": clean_prompt,
        "id": str(uuid.uuid4()),
    })

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

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resp_text,
                    "sources": sources,
                    "ticket_id": ticket_id,
                    "id": str(uuid.uuid4()),
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


with st.sidebar:
    st.subheader("Session Info")
    st.text(f"Session ID:\n{st.session_state.session_id}")
    if st.button("New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.audio_cache = {}
        st.session_state.voice_transcript = ""
        st.session_state.last_transcribed_audio_id = None
        st.rerun()


# Voice input section with audio recorder and editable transcript
# Keep expanded so recording and reviewing is always directly visible
with st.expander("🎙️ Voice Input (Record Question)", expanded=True):
    st.caption("Record your question using the microphone. You will be able to review and edit the transcribed text before submitting to the agent.")
    audio_file = st.audio_input("Record audio question", key="mic_recorder")

    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        if not audio_bytes or len(audio_bytes) < 100:
            st.warning("Recorded audio is empty or too short. Please speak into your microphone.")
        else:
            audio_hash = hashlib.sha256(audio_bytes).hexdigest()
            if st.session_state.last_transcribed_audio_id != audio_hash:
                with st.spinner("Transcribing audio..."):
                    try:
                        content_type = getattr(audio_file, "type", None) or "audio/wav"
                        filename = getattr(audio_file, "name", None) or "recording.wav"
                        files = {"file": (filename, audio_bytes, content_type)}
                        with httpx.Client(timeout=45.0) as client:
                            t_res = client.post(f"{API_BASE_URL}/voice/transcribe", files=files)
                            if t_res.status_code == 200:
                                data = t_res.json()
                                transcript_text = data.get("transcript", "").strip()
                                if transcript_text:
                                    st.session_state.voice_transcript = transcript_text
                                    st.session_state.last_transcribed_audio_id = audio_hash
                                    st.rerun()

                                else:
                                    st.warning("No speech could be detected in the recording. Please try speaking clearly.")
                            else:
                                err = t_res.json().get("detail") or t_res.json().get("error") or t_res.text
                                st.warning(f"Transcription notice: {err}")
                    except Exception as exc:
                        st.error(f"Transcription request failed: {exc}")

    if st.session_state.voice_transcript:
        st.write("##### Review & Edit Transcript:")
        draft_key = f"transcript_edit_{st.session_state.last_transcribed_audio_id}"
        edited_transcript = st.text_area(
            "Transcribed speech (editable):",
            value=st.session_state.voice_transcript,
            key=draft_key,
            label_visibility="collapsed",
            help="Review or edit the text. Click 'Send Voice Question' to submit.",
        )
        col_send, col_discard = st.columns([2, 1])
        with col_send:
            if st.button("📤 Send Voice Question", key="send_voice_btn", type="primary", use_container_width=True):
                q = edited_transcript.strip()
                if q:
                    st.session_state.voice_transcript = ""
                    process_user_query(q)
                    st.rerun()
        with col_discard:
            if st.button("Discard", key="discard_transcript_btn", use_container_width=True):
                st.session_state.voice_transcript = ""
                st.rerun()



# Display chat history
for idx, message in enumerate(st.session_state.messages):
    msg_role = message.get("role", "user")
    with st.chat_message(msg_role):
        st.write(message.get("content", ""))
        if msg_role == "assistant":
            render_assistant_extra(message)
            msg_id = message.get("id") or f"msg_{idx}"
            if st.button("🔊 Listen", key=f"listen_{msg_id}"):
                if msg_id not in st.session_state.audio_cache:
                    with st.spinner("Generating speech..."):
                        try:
                            with httpx.Client(timeout=30.0) as client:
                                s_res = client.post(
                                    f"{API_BASE_URL}/voice/synthesize",
                                    json={"message_id": msg_id, "text": message["content"]},
                                )
                                if s_res.status_code == 200:
                                    ct = s_res.headers.get("content-type", "audio/mpeg")
                                    st.session_state.audio_cache[msg_id] = {
                                        "bytes": s_res.content,
                                        "media_type": ct,
                                    }
                                else:
                                    st.warning(f"Speech playback unavailable ({s_res.status_code}): {s_res.text}")
                        except Exception as exc:
                            st.warning(f"Voice playback failed: {exc}")

            if msg_id in st.session_state.audio_cache:
                cached = st.session_state.audio_cache[msg_id]
                st.audio(cached["bytes"], format=cached["media_type"], autoplay=False)

# Typed chat input remains fully functional
if prompt := st.chat_input("How can we help?"):
    clean_prompt = prompt.strip()
    if clean_prompt:
        process_user_query(clean_prompt)
        st.rerun()


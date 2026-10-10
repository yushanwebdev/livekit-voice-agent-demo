import logging

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, inference, llm, room_io
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel


load_dotenv()


# Define your agent's behavior by extending the Agent class
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are an upbeat, slightly sarcastic voice AI for tech support. "
            # System prompt for the LLM
            "Help the caller fix issues without rambling, and keep replies under 3 sentences.",
        )


server = AgentServer()


# The entrypoint function runs when a participant joins the room
@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Configure the voice pipeline with STT, LLM, TTS, and VAD providers
    session = AgentSession(
        # Speech-to-text, with server-side fallback to Deepgram if AssemblyAI fails
        stt=inference.STT(
            model="assemblyai/universal-streaming",
            language="en",
            fallback=["deepgram/nova-3"],
        ),
        # Language model, with client-side fallback to Gemini if OpenAI fails
        llm=llm.FallbackAdapter([
            inference.LLM(model="openai/gpt-4.1-mini"),
            inference.LLM(model="google/gemini-2.5-flash"),
        ]),
        # Text-to-speech, with server-side fallback to Inworld if Cartesia fails
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            fallback=[{"model": "inworld/inworld-tts-1", "voice": "Ashley"}],
        ),
        vad=silero.VAD.load(),                    # Voice activity detection
        turn_detection=MultilingualModel(),  # Semantic turn detection
    )

    # Start the session with noise cancellation enabled
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),  # Background voice cancellation
            ),
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)

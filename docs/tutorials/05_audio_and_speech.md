# 05 — Audio, Music & Speech

## Music generation

```python
import neura_x as nx

music = nx.AudioGen(architecture="autoregressive_transformer",
                    conceptual_params=1_500_000_000,
                    sample_rate=32000, duration_seconds=30)

nx.TrainingEngine(music).train_from_scratch(dataset="music/")
track = music.generate("Upbeat Afrobeat rhythm with guitar")
track.save("afrobeat.wav")
```

---

## Text-to-Speech (Including Swahili)

```python
tts = nx.SpeechSynthesis(architecture="vits",
                         conceptual_params=500_000_000,
                         languages=["en", "sw"])
nx.TrainingEngine(tts).train(dataset="speech_sw/")
tts.synthesize("Habari yako? Neura-X hapa.").save("greeting.wav")
```

## Speech-to-Text

```python
stt = nx.SpeechRecognition(architecture="whisper_style", languages=["en","sw"])
text = stt.transcribe(nx.utils.load_audio("meeting.wav"))
```

A student in Nairobi can now give a machine a voice in their own language on a $500 laptop, with no API fees, forever.


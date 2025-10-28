# GrokipediaSimulator 🚀

AI-powered knowledge simulator inspired by Grok & Wikipedia. Bundles 10+ prototypes for querying, validating, forecasting, and ethical sims. Built for truth-seeking workflows—recursive dual reasoning, VAE tracking, Monte Carlo ROI.

## Quickstart
1. Clone: `git clone https://github.com/brianjameschapman-tech/Grokipedia-Simulator.git`
2. Install: `pip install -r requirements.txt`
3. Run: `python main.py "Tesla Vote" --mode recursive --persist`

### Modes
- **Single**: One-shot sim.
- **Recursive**: Evolves topics over depth; plots divergence.
- **Continuous**: Interactive loop (input topics till 'quit').

### Web Deployment
- Local: `streamlit run app.py`
- Free Host: [Streamlit Cloud](https://streamlit.io/cloud) > Link repo > Deploy `app.py`.

## Features
- **Core Pipeline**: Query → Cite → Mindmap → Dual Validate (SymPy/Torch) → Narrate → Viz → Edit → Translate → Analytics.
- **Protos**: VAE (Bayesian), MC Forecaster (NPV/ROI), Ethics Sim (opt-in ROI).
- **xAI Ties**: API stubs (https://x.ai/api); monetization hints.

## Testing
`pytest tests/` 

## License
MIT - Fork away!

Built by @RhetoricNoob. Issues? Open one.
